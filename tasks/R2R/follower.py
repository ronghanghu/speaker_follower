''' Agents: stop/random/shortest/seq2seq  '''

import json
import sys
import numpy as np
import random
from collections import namedtuple

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as D

from utils import vocab_pad_idx, vocab_eos_idx, flatten, structured_map, try_cuda

#from env import FOLLOWER_MODEL_ACTIONS, FOLLOWER_ENV_ACTIONS, IGNORE_ACTION_INDEX, LEFT_ACTION_INDEX, RIGHT_ACTION_INDEX, START_ACTION_INDEX, END_ACTION_INDEX, FORWARD_ACTION_INDEX, index_action_tuple

InferenceState = namedtuple("InferenceState", "prev_inference_state, world_state, observation, flat_index, last_action, last_action_embedding, action_count, score, h_t, c_t")

def backchain_inference_states(last_inference_state):
    states = []
    observations = []
    actions = []
    inf_state = last_inference_state
    scores = []
    last_score = None
    while inf_state is not None:
        states.append(inf_state.world_state)
        observations.append(inf_state.observation)
        actions.append(inf_state.last_action)
        if last_score is not None:
            scores.append(last_score - inf_state.score)
        last_score = inf_state.score
        inf_state = inf_state.prev_inference_state
    scores.append(last_score)
    return list(reversed(states)), list(reversed(observations)), list(reversed(actions))[1:], list(reversed(scores))[1:] # exclude start action

def batch_instructions_from_encoded(encoded_instructions, max_length, reverse=False, sort=False):
    # encoded_instructions: list of lists of token indices (should not be padded, or contain BOS or EOS tokens)
    #seq_tensor = np.array(encoded_instructions)
    # make sure pad does not start any sentence
    num_instructions = len(encoded_instructions)
    seq_tensor = np.full((num_instructions, max_length), vocab_pad_idx)
    seq_lengths = []
    for i, inst in enumerate(encoded_instructions):
        if len(inst) > 0:
            assert inst[-1] != vocab_eos_idx
        if reverse:
            inst = inst[::-1]
        inst = np.concatenate((inst, [vocab_eos_idx]))
        inst = inst[:max_length]
        seq_tensor[i,:len(inst)] = inst
        seq_lengths.append(len(inst))

    seq_tensor = torch.from_numpy(seq_tensor)
    if sort:
        seq_lengths, perm_idx = torch.from_numpy(np.array(seq_lengths)).sort(0, True)
        seq_lengths = list(seq_lengths)
        seq_tensor = seq_tensor[perm_idx]

    mask = (seq_tensor == vocab_pad_idx)[:, :max(seq_lengths)]

    ret_tp = try_cuda(Variable(seq_tensor, requires_grad=False).long()), \
             try_cuda(mask.byte()), \
             seq_lengths
    if sort:
        ret_tp = ret_tp + (list(perm_idx),)
    return ret_tp

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        results = {}
        for key, item in self.results.items():
            results[key] = {
                'instr_id': item['instr_id'],
                'trajectory': item['trajectory'],
            }
        with open(self.results_path, 'w') as f:
            json.dump(results, f)

    def rollout(self):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}

        # We rely on env showing the entire batch before repeating anything
        #print 'Testing %s' % self.__class__.__name__
        looped = False
        rollout_scores = []
        beam_10_scores = []
        while True:
            rollout_results = self.rollout()
            # if self.feedback == 'argmax':
                # beam_results = self.beam_search(1, load_next_minibatch=False)
                # assert len(rollout_results) == len(beam_results)
                # for rollout_traj, beam_trajs in zip(rollout_results, beam_results):
                #     assert rollout_traj['instr_id'] == beam_trajs[0]['instr_id']
                #     assert rollout_traj['trajectory'] == beam_trajs[0]['trajectory']
                #     assert np.allclose(rollout_traj['score'], beam_trajs[0]['score'])
                # print("passed check: beam_search with beam_size=1")
                #
                # self.env.set_beam_size(10)
                # beam_results = self.beam_search(10, load_next_minibatch=False)
                # assert len(rollout_results) == len(beam_results)
                # for rollout_traj, beam_trajs in zip(rollout_results, beam_results):
                #     rollout_score = rollout_traj['score']
                #     rollout_scores.append(rollout_score)
                #     beam_score = beam_trajs[0]['score']
                #     beam_10_scores.append(beam_score)
                #     # assert rollout_score <= beam_score
                # self.env.set_beam_size(1)
                # # print("passed check: beam_search with beam_size=10")
            # if self.feedback == 'teacher' and self.beam_size == 1:
            #     rollout_loss = self.loss
            #     path_obs, path_actions, encoded_instructions = self.env.gold_obs_actions_and_instructions(self.episode_len, load_next_minibatch=False)
            #     for i in range(len(rollout_results)):
            #         assert rollout_results[i]['actions'] == path_actions[i]
            #         assert [o1['viewpoint'] == o2['viewpoint']
            #                 for o1, o2 in zip(rollout_results[i]['observations'], path_obs[i])]
            #     trajs, loss = self._score_obs_actions_and_instructions(path_obs, path_actions, encoded_instructions)
            #     for traj, rollout in zip(trajs, rollout_results):
            #         assert traj['instr_id'] == rollout['instr_id']
            #         assert traj['actions'] == rollout['actions']
            #         assert np.allclose(traj['score'], rollout['score'])
            #     assert np.allclose(rollout_loss.data[0], loss.data[0])
            #     print('passed score test')

            for result in rollout_results:
                if result['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[result['instr_id']] = result
            if looped:
                break
        # if self.feedback == 'argmax':
        #     print("avg rollout score: ", np.mean(rollout_scores))
        #     print("avg beam 10 score: ", np.mean(beam_10_scores))
        return self.results

def path_element_from_observation(ob):
    return (ob['viewpoint'], ob['heading'], ob['elevation'])

class StopAgent(BaseAgent):
    ''' An agent that doesn't move! '''

    def rollout(self):
        world_states = self.env.reset()
        obs = self.env.observe(world_states)
        traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob) ]
        } for ob in obs]
        return traj


class RandomAgent(BaseAgent):
    ''' An agent that picks a random direction then tries to go straight for
        five viewpoint steps and then stops. '''

    def rollout(self):
        world_states = self.env.reset()
        obs = self.env.observe(world_states)
        traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob)]
        } for ob in obs]
        ended = [False] * len(obs)

        self.steps = [0] * len(obs)
        for t in range(6):
            actions = []
            for i, ob in enumerate(obs):
                if self.steps[i] >= 5:
                    actions.append(0)  # do nothing, i.e. end
                    ended[i] = True
                elif self.steps[i] == 0:
                    a = np.random.randint(len(ob['adj_loc_list']) - 1) + 1
                    actions.append(a)  # choose a random adjacent loc
                    self.steps[i] += 1
                else:
                    assert len(ob['adj_loc_list']) > 1
                    actions.append(1)  # go forward
                    self.steps[i] += 1
            world_states = self.env.step(world_states, actions, obs)
            obs = self.env.observe(world_states)
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['trajectory'].append(path_element_from_observation(ob))
        return traj

class ShortestAgent(BaseAgent):
    ''' An agent that always takes the shortest path to goal. '''

    def rollout(self):
        world_states = self.env.reset()
        #obs = self.env.observe(world_states)
        all_obs, all_actions = self.env.shortest_paths_to_goals(world_states, 20)
        return [
            {
                'instr_id': obs[0]['instr_id'],
                # end state will appear twice because stop action is a no-op, so exclude it
                'trajectory': [path_element_from_observation(ob) for ob in obs[:-1]]
            }
            for obs in all_obs
        ]

class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    # env_actions = FOLLOWER_ENV_ACTIONS
    # start_index = START_ACTION_INDEX
    # ignore_index = IGNORE_ACTION_INDEX
    # forward_index = FORWARD_ACTION_INDEX
    # end_index = END_ACTION_INDEX
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, encoder, decoder, episode_len=10, beam_size=1, reverse_instruction=True, max_instruction_length=80):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.encoder = encoder
        self.decoder = decoder
        self.episode_len = episode_len
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.beam_size = beam_size
        self.reverse_instruction = reverse_instruction
        self.max_instruction_length = max_instruction_length

    # @staticmethod
    # def n_inputs():
    #     return len(FOLLOWER_MODEL_ACTIONS)
    #
    # @staticmethod
    # def n_outputs():
    #     return len(FOLLOWER_MODEL_ACTIONS)-2 # Model doesn't output start or ignore

    def _feature_variables(self, obs, beamed=False):
        ''' Extract precomputed features into variable. '''
        feature_lists = list(zip(*[ob['feature'] for ob in (flatten(obs) if beamed else obs)]))
        assert len(feature_lists) == len(self.env.image_features_list)
        batched = []
        for featurizer, feature_list in zip(self.env.image_features_list, feature_lists):
            batched.append(featurizer.batch_features(feature_list))
        return batched

    def _action_variable(self, obs):
        # get the maximum number of actions of all sample in this batch
        max_num_a = -1
        for i, ob in enumerate(obs):
            max_num_a = max(max_num_a, len(ob['adj_loc_list']))

        is_valid = np.zeros((len(obs), max_num_a), np.float32)
        action_embedding_dim = obs[0]['action_embedding'].shape[-1]
        action_embeddings = np.zeros(
            (len(obs), max_num_a, action_embedding_dim),
            dtype=np.float32)
        for i, ob in enumerate(obs):
            adj_loc_list = ob['adj_loc_list']
            num_a = len(adj_loc_list)
            is_valid[i, 0:num_a] = 1.
            for n_a, adj_dict in enumerate(adj_loc_list):
                action_embeddings[i, :num_a, :] = ob['action_embedding']
        return (
            Variable(torch.from_numpy(action_embeddings), requires_grad=False).cuda(),
            Variable(torch.from_numpy(is_valid), requires_grad=False).cuda(),
            is_valid)

    def _teacher_action(self, obs, ended):
        ''' Extract teacher actions into variable. '''
        a = torch.LongTensor(len(obs))
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            a[i] = ob['teacher'] if not ended[i] else -1
        return try_cuda(Variable(a, requires_grad=False))

    def _proc_batch(self, obs, beamed=False):
        encoded_instructions = [ob['instr_encoding'] for ob in (flatten(obs) if beamed else obs)]
        return batch_instructions_from_encoded(encoded_instructions, self.max_instruction_length, reverse=self.reverse_instruction)

    def rollout(self):
        if self.beam_size == 1:
            return self._rollout_with_loss()
        else:
            assert self.beam_size >= 1
            beams = self.beam_search(self.beam_size)
            return [beam[0] for beam in beams]

    def _score_obs_actions_and_instructions(self, path_obs, path_actions, encoded_instructions):
        batch_size = len(path_obs)
        assert len(path_actions) == batch_size
        assert len(encoded_instructions) == batch_size
        for path_o, path_a in zip(path_obs, path_actions):
            assert len(path_o) == len(path_a) + 1

        seq, seq_mask, seq_lengths, perm_indices = \
            batch_instructions_from_encoded(
                encoded_instructions, self.max_instruction_length,
                reverse=self.reverse_instruction, sort=True)
        loss = 0

        ctx, h_t, c_t = self.encoder(seq, seq_lengths)
        u_t_prev = self.decoder.u_begin.expand(batch_size, -1)  # init action
        ended = np.array([False] * batch_size)
        sequence_scores = try_cuda(torch.zeros(batch_size))

        traj = [{
            'instr_id': path_o[0]['instr_id'],
            'trajectory': [path_element_from_observation(path_o[0])],
            'actions': [],
            'scores': [],
            'observations': [path_o[0]],
            'instr_encoding': path_o[0]['instr_encoding']
        } for path_o in path_obs]

        obs = None
        for t in range(self.episode_len):
            next_obs = []
            next_target_list = []
            for perm_index, src_index in enumerate(perm_indices):
                path_o = path_obs[src_index]
                path_a = path_actions[src_index]
                if t < len(path_a):
                    next_target_list.append(path_a[t])
                    next_obs.append(path_o[t])
                else:
                    next_target_list.append(-1)
                    next_obs.append(obs[perm_index])

            obs = next_obs

            target = try_cuda(Variable(torch.LongTensor(next_target_list), requires_grad=False))

            f_t_list = self._feature_variables(obs) # Image features from obs
            all_u_t, is_valid, _ = self._action_variable(obs)

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'
            h_t, c_t, alpha, logit, alpha_v = self.decoder(
                u_t_prev, all_u_t, f_t_list[0], h_t, c_t, ctx, seq_mask)

            # Mask outputs of invalid actions
            logit[is_valid == 0] = -float('inf')

            # Supervised training
            loss += self.criterion(logit, target)

            # Determine next model inputs
            a_t = torch.clamp(target, min=0)  # teacher forcing
            # update the previous action
            u_t_prev = all_u_t[np.arange(batch_size), a_t, :].detach()

            action_scores = -F.cross_entropy(logit, target, ignore_index=-1, reduce=False).data
            sequence_scores += action_scores

            # Save trajectory output
            for perm_index, src_index in enumerate(perm_indices):
                ob = obs[perm_index]
                if not ended[perm_index]:
                    traj[src_index]['trajectory'].append(path_element_from_observation(ob))
                    traj[src_index]['score'] = float(sequence_scores[perm_index])
                    traj[src_index]['scores'].append(action_scores[perm_index])
                    traj[src_index]['actions'].append(a_t.data[perm_index])
                    # traj[src_index]['observations'].append(ob)

            # Update ended list
            for i in range(batch_size):
                action_idx = a_t[i].data[0]
                if action_idx == 0:
                    ended[i] = True

            # Early exit if all ended
            if ended.all():
                break

        return traj, loss

    def _rollout_with_loss(self):
        initial_world_states = self.env.reset(sort=True)
        initial_obs = self.env.observe(initial_world_states)
        initial_obs = np.array(initial_obs)
        batch_size = len(initial_obs)

        # get mask and lengths
        seq, seq_mask, seq_lengths = self._proc_batch(initial_obs)

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        # TODO consider not feeding this into the decoder, and just using attention

        self.loss = 0

        feedback = self.feedback

        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob)],
            'actions': [],
            'scores': [],
            'observations': [ob],
            'instr_encoding': ob['instr_encoding']
        } for ob in initial_obs]

        obs = initial_obs
        world_states = initial_world_states

        # Initial action
        u_t_prev = self.decoder.u_begin.expand(batch_size, -1)  # init action
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        env_action = [None] * batch_size
        sequence_scores = try_cuda(torch.zeros(batch_size))
        for t in range(self.episode_len):
            f_t_list = self._feature_variables(obs) # Image features from obs
            all_u_t, is_valid, _ = self._action_variable(obs)

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'
            h_t, c_t, alpha, logit, alpha_v = self.decoder(
                u_t_prev, all_u_t, f_t_list[0], h_t, c_t, ctx, seq_mask)

            # Mask outputs of invalid actions
            logit[is_valid == 0] = -float('inf')

            # Supervised training
            target = self._teacher_action(obs, ended)
            self.loss += self.criterion(logit, target)

            # Determine next model inputs
            if feedback == 'teacher':
                # turn -1 (ignore) to 0 (stop) so that the action is executable
                a_t = torch.clamp(target, min=0)
            elif feedback == 'argmax':
                _,a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
            elif feedback == 'sample':
                probs = F.softmax(logit, dim=1)    # sampling an action from model
                # Further mask probs where agent can't move forward
                # Note input to `D.Categorical` does not have to sum up to 1
                # http://pytorch.org/docs/stable/torch.html#torch.multinomial
                probs[is_valid == 0] = 0.
                m = D.Categorical(probs)
                a_t = m.sample()
            else:
                sys.exit('Invalid feedback option')

            # update the previous action
            u_t_prev = all_u_t[np.arange(batch_size), a_t, :].detach()

            action_scores = -F.cross_entropy(logit, a_t, ignore_index=-1, reduce=False).data
            sequence_scores += action_scores

            # dfried: I changed this so that the ended list is updated afterward; this causes <end> to be added as the last action, along with its score, and the final world state will be duplicated (to more closely match beam search)
            # Make environment action
            for i in range(batch_size):
                action_idx = a_t[i].data[0]
                env_action[i] = action_idx

            world_states = self.env.step(world_states, env_action, obs)
            obs = self.env.observe(world_states)
            # print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, world_states[0], a_t.data[0], sequence_scores[0]))

            # Save trajectory output
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['trajectory'].append(path_element_from_observation(ob))
                    traj[i]['score'] = sequence_scores[i]
                    traj[i]['scores'].append(action_scores[i])
                    traj[i]['actions'].append(a_t.data[i])
                    traj[i]['observations'].append(ob)

            # Update ended list
            for i in range(batch_size):
                action_idx = a_t[i].data[0]
                if action_idx == 0:
                    ended[i] = True

            # Early exit if all ended
            if ended.all():
                break

        #self.losses.append(self.loss.data[0] / self.episode_len)
        # shouldn't divide by the episode length because of masking
        self.losses.append(self.loss.data[0])
        return traj

    def beam_search(self, beam_size, load_next_minibatch=True, mask_undo=False):
        assert self.env.beam_size >= beam_size
        world_states = self.env.reset(sort=True, beamed=True, load_next_minibatch=load_next_minibatch)
        obs = self.env.observe(world_states, beamed=True)
        batch_size = len(world_states)

        # get mask and lengths
        seq, seq_mask, seq_lengths = self._proc_batch(obs, beamed=True)

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        completed = []
        for _ in range(batch_size):
            completed.append([])

        beams = [
            [InferenceState(prev_inference_state=None,
                            world_state=ws[0],
                            observation=o[0],
                            flat_index=i,
                            last_action=-1,
                            last_action_embedding=self.decoder.u_begin.view(-1),
                            action_count=0,
                            score=0.0, h_t=None, c_t=None)]
            for i, (ws, o) in enumerate(zip(world_states, obs))
        ]

        # Do a sequence rollout and calculate the loss
        for t in range(self.episode_len):
            flat_indices = []
            beam_indices = []
            u_t_list = []
            for beam_index, beam in enumerate(beams):
                for inf_state in beam:
                    beam_indices.append(beam_index)
                    flat_indices.append(inf_state.flat_index)
                    u_t_list.append(inf_state.last_action_embedding)

            u_t_prev = torch.stack(u_t_list, dim=0)
            assert len(u_t_prev.shape) == 2
            flat_obs = flatten(obs)
            f_t_list = self._feature_variables(flat_obs) # Image features from obs
            all_u_t, is_valid, is_valid_numpy = self._action_variable(flat_obs)

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'
            h_t, c_t, alpha, logit, alpha_v = self.decoder(
                u_t_prev, all_u_t, f_t_list[0], h_t[flat_indices], c_t[flat_indices], ctx[beam_indices], seq_mask[beam_indices])

            # Mask outputs of invalid actions
            logit[is_valid == 0] = -float('inf')
            # # Mask outputs where agent can't move forward
            # no_forward_mask = [len(ob['navigableLocations']) <= 1 for ob in flat_obs]

            if mask_undo:
                masked_logit = logit.clone()
            else:
                masked_logit = logit

            log_probs = F.log_softmax(logit, dim=1).data

            # force ending if we've reached the max time steps
            # if t == self.episode_len - 1:
            #     action_scores = log_probs[:,self.end_index].unsqueeze(-1)
            #     action_indices = torch.from_numpy(np.full((log_probs.size()[0], 1), self.end_index))
            # else:
            #action_scores, action_indices = log_probs.topk(min(beam_size, logit.size()[1]), dim=1)
            _, action_indices = masked_logit.data.topk(min(beam_size, logit.size()[1]), dim=1)
            action_scores = log_probs.gather(1, action_indices)
            assert action_scores.size() == action_indices.size()

            start_index = 0
            new_beams = []
            assert len(beams) == len(world_states)
            all_successors = []
            for beam_index, (beam, beam_world_states, beam_obs) in enumerate(zip(beams, world_states, obs)):
                successors = []
                end_index = start_index + len(beam)
                assert len(beam_world_states) == len(beam)
                assert len(beam_obs) == len(beam)
                if beam:
                    for inf_index, (inf_state, world_state, ob, action_score_row, action_index_row) in \
                            enumerate(zip(beam, beam_world_states, beam_obs, action_scores[start_index:end_index], action_indices[start_index:end_index])):
                        flat_index = start_index + inf_index
                        for action_score, action_index in zip(action_score_row, action_index_row):
                            if is_valid_numpy[flat_index, action_index] == 0:
                                continue
                            successors.append(
                                InferenceState(prev_inference_state=inf_state,
                                               world_state=world_state, # will be updated later after successors are pruned
                                               observation=ob, # will be updated later after successors are pruned
                                               flat_index=flat_index,
                                               last_action=action_index,
                                               last_action_embedding=all_u_t[flat_index, action_index].detach(),
                                               action_count=inf_state.action_count + 1,
                                               score=float(inf_state.score + action_score), h_t=None, c_t=None)
                            )
                start_index = end_index
                successors = sorted(successors, key=lambda t: t.score, reverse=True)[:beam_size]
                all_successors.append(successors)

            successor_world_states = [
                [inf_state.world_state for inf_state in successors]
                for successors in all_successors
            ]

            successor_env_actions = [
                [inf_state.last_action for inf_state in successors]
                for successors in all_successors
            ]

            successor_last_obs = [
                [inf_state.observation for inf_state in successors]
                for successors in all_successors
            ]

            successor_world_states = self.env.step(successor_world_states, successor_env_actions, successor_last_obs, beamed=True)
            successor_obs = self.env.observe(successor_world_states, beamed=True)

            all_successors = structured_map(lambda inf_state, world_state, obs: inf_state._replace(world_state=world_state, observation=obs),
                                   all_successors, successor_world_states, successor_obs, nested=True)

            # if all_successors[0]:
            #     print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, all_successors[0][0].world_state, all_successors[0][0].last_action, all_successors[0][0].score))

            for beam_index, successors in enumerate(all_successors):
                new_beam = []
                for successor in successors:
                    if successor.last_action == 0 or t == self.episode_len - 1:
                        completed[beam_index].append(successor)
                    else:
                        new_beam.append(successor)
                if len(completed[beam_index]) >= beam_size:
                    new_beam = []
                new_beams.append(new_beam)

            beams = new_beams

            world_states = [
                [inf_state.world_state for inf_state in beam]
                for beam in beams
            ]

            obs = [
                [inf_state.observation for inf_state in beam]
                for beam in beams
            ]

            # Early exit if all ended
            if not any(beam for beam in beams):
                break

        trajs = []

        for this_completed in completed:
            assert this_completed
            this_trajs = []
            for inf_state in sorted(this_completed, key=lambda t: t.score, reverse=True)[:beam_size]:
                path_states, path_observations, path_actions, path_scores = backchain_inference_states(inf_state)
                # this will have messed-up headings for (at least some) starting locations because of
                # discretization, so read from the observations instead
                ## path = [(obs.viewpointId, state.heading, state.elevation)
                ##         for state in path_states]
                trajectory = [path_element_from_observation(ob) for ob in path_observations]
                this_trajs.append({
                    'instr_id': path_observations[0]['instr_id'],
                    'instr_encoding': path_observations[0]['instr_encoding'],
                    'trajectory': trajectory,
                    'observations': path_observations,
                    'actions': path_actions,
                    'score': inf_state.score,
                    'scores': path_scores,
                })
            trajs.append(this_trajs)
        return trajs

    def state_factored_search(self, completion_size, successor_size, load_next_minibatch=True, mask_undo=False):
        assert self.env.beam_size >= successor_size
        world_states = self.env.reset(sort=True, beamed=True, load_next_minibatch=load_next_minibatch)
        initial_obs = self.env.observe(world_states, beamed=True)
        batch_size = len(world_states)

        # get mask and lengths
        seq, seq_mask, seq_lengths = self._proc_batch(initial_obs, beamed=True)

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        completed = []
        completed_holding = []
        for _ in range(batch_size):
            completed.append({})
            completed_holding.append({})

        state_cache = [
            {ws[0]: (InferenceState(prev_inference_state=None,
                                    world_state=ws[0],
                                    observation=o[0],
                                    flat_index=None,
                                    last_action=-1,
                                    last_action_embedding=self.decoder.u_begin.view(-1),
                                    action_count=0,
                                    score=0.0, h_t=h_t[i], c_t=c_t[i]), True)}
            for i, (ws, o) in enumerate(zip(world_states, initial_obs))
        ]

        beams = [[inf_state for world_state, (inf_state, expanded) in sorted(instance_cache.items())]
                 for instance_cache in state_cache] # sorting is a noop here since each instance_cache should only contain one

        # Do a sequence rollout and calculate the loss
        while any(len(comp) < completion_size for comp in completed):
            beam_indices = []
            u_t_list = []
            h_t_list = []
            c_t_list = []
            flat_obs = []
            for beam_index, beam in enumerate(beams):
                for inf_state in beam:
                    beam_indices.append(beam_index)
                    u_t_list.append(inf_state.last_action_embedding)
                    h_t_list.append(inf_state.h_t.unsqueeze(0))
                    c_t_list.append(inf_state.c_t.unsqueeze(0))
                    flat_obs.append(inf_state.observation)

            u_t_prev = torch.stack(u_t_list, dim=0)
            assert len(u_t_prev.shape) == 2
            f_t_list = self._feature_variables(flat_obs) # Image features from obs
            all_u_t, is_valid, is_valid_numpy = self._action_variable(flat_obs)
            h_t = torch.cat(h_t_list, dim=0)
            c_t = torch.cat(c_t_list, dim=0)

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'
            h_t, c_t, alpha, logit, alpha_v = self.decoder(
                u_t_prev, all_u_t, f_t_list[0], h_t, c_t, ctx[beam_indices], seq_mask[beam_indices])

            # Mask outputs of invalid actions
            logit[is_valid == 0] = -float('inf')
            # # Mask outputs where agent can't move forward
            # no_forward_mask = [len(ob['navigableLocations']) <= 1 for ob in flat_obs]

            if mask_undo:
                masked_logit = logit.clone()
            else:
                masked_logit = logit

            log_probs = F.log_softmax(logit, dim=1).data

            # force ending if we've reached the max time steps
            # if t == self.episode_len - 1:
            #     action_scores = log_probs[:,self.end_index].unsqueeze(-1)
            #     action_indices = torch.from_numpy(np.full((log_probs.size()[0], 1), self.end_index))
            # else:
            #_, action_indices = masked_logit.data.topk(min(successor_size, logit.size()[1]), dim=1)
            _, action_indices = masked_logit.data.topk(logit.size()[1], dim=1) # todo: fix this
            action_scores = log_probs.gather(1, action_indices)
            assert action_scores.size() == action_indices.size()

            start_index = 0
            assert len(beams) == len(world_states)
            all_successors = []
            for beam_index, (beam, beam_world_states) in enumerate(zip(beams, world_states)):
                successors = []
                end_index = start_index + len(beam)
                assert len(beam_world_states) == len(beam)
                if beam:
                    for inf_index, (inf_state, world_state, action_score_row) in \
                            enumerate(zip(beam, beam_world_states, log_probs[start_index:end_index])):
                        flat_index = start_index + inf_index
                        for action_index, action_score in enumerate(action_score_row):
                            if is_valid_numpy[flat_index, action_index] == 0:
                                continue
                            successors.append(
                                InferenceState(prev_inference_state=inf_state,
                                               world_state=world_state, # will be updated later after successors are pruned
                                               observation=flat_obs[flat_index], # will be updated later after successors are pruned
                                               flat_index=None,
                                               last_action=action_index,
                                               last_action_embedding=all_u_t[flat_index, action_index].detach(),
                                               action_count=inf_state.action_count + 1,
                                               score=inf_state.score + action_score, h_t=h_t[flat_index], c_t=c_t[flat_index])
                            )
                start_index = end_index
                successors = sorted(successors, key=lambda t: t.score, reverse=True)
                all_successors.append(successors)

            successor_world_states = [
                [inf_state.world_state for inf_state in successors]
                for successors in all_successors
            ]

            successor_env_actions = [
                [inf_state.last_action for inf_state in successors]
                for successors in all_successors
            ]

            successor_last_obs = [
                [inf_state.observation for inf_state in successors]
                for successors in all_successors
            ]

            successor_world_states = self.env.step(successor_world_states, successor_env_actions, successor_last_obs, beamed=True)

            all_successors = structured_map(lambda inf_state, world_state: inf_state._replace(world_state=world_state),
                                            all_successors, successor_world_states, nested=True)

            # if all_successors[0]:
            #     print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, all_successors[0][0].world_state, all_successors[0][0].last_action, all_successors[0][0].score))

            assert len(all_successors) == len(state_cache)

            new_beams = []

            for beam_index, (successors, instance_cache) in enumerate(zip(all_successors, state_cache)):
                # early stop if we've already built a sizable completion list
                instance_completed = completed[beam_index]
                instance_completed_holding = completed_holding[beam_index]
                if len(instance_completed) >= completion_size:
                    new_beams.append([])
                    continue
                for successor in successors:
                    ws = successor.world_state
                    if successor.last_action == 0 or successor.action_count == self.episode_len:
                        if ws not in instance_completed_holding or instance_completed_holding[ws][0].score < successor.score:
                            instance_completed_holding[ws] = (successor, False)
                    else:
                        if ws not in instance_cache or instance_cache[ws][0].score < successor.score:
                            instance_cache[ws] = (successor, False)

                # third value: did this come from completed_holding?
                uncompleted_to_consider = ((ws, inf_state, False) for (ws, (inf_state, expanded)) in instance_cache.items() if not expanded)
                completed_to_consider = ((ws, inf_state, True) for (ws, (inf_state, expanded)) in instance_completed_holding.items() if not expanded)
                import itertools
                import heapq
                to_consider = itertools.chain(uncompleted_to_consider, completed_to_consider)
                ws_and_inf_states = heapq.nlargest(successor_size, to_consider, key=lambda pair: pair[1].score)

                new_beam = []
                for ws, inf_state, is_completed in ws_and_inf_states:
                    if is_completed:
                        assert instance_completed_holding[ws] == (inf_state, False)
                        instance_completed_holding[ws] = (inf_state, True)
                        if ws not in instance_completed or instance_completed[ws].score < inf_state.score:
                            instance_completed[ws] = inf_state
                    else:
                        instance_cache[ws] = (inf_state, True)
                        new_beam.append(inf_state)

                if len(instance_completed) >= completion_size:
                    new_beams.append([])
                else:
                    new_beams.append(new_beam)

            beams = new_beams

            # Early exit if all ended
            if not any(beam for beam in beams):
                break

            world_states = [
                [inf_state.world_state for inf_state in beam]
                for beam in beams
            ]
            successor_obs = self.env.observe(world_states, beamed=True)
            beams = structured_map(lambda inf_state, obs: inf_state._replace(observation=obs),
                                   beams, successor_obs, nested=True)

        completed_list = []
        for this_completed in completed:
            completed_list.append(sorted(this_completed.values(), key=lambda t: t.score, reverse=True)[:completion_size])
        completed_ws = [
            [inf_state.world_state for inf_state in comp_l]
            for comp_l in completed_list
        ]
        completed_obs = self.env.observe(completed_ws, beamed=True)
        completed_list = structured_map(lambda inf_state, obs: inf_state._replace(observation=obs),
                                        completed_list, completed_obs, nested=True)

        trajs = []
        for this_completed in completed_list:
            assert this_completed
            this_trajs = []
            for inf_state in this_completed:
                path_states, path_observations, path_actions, path_scores = backchain_inference_states(inf_state)
                # this will have messed-up headings for (at least some) starting locations because of
                # discretization, so read from the observations instead
                ## path = [(obs.viewpointId, state.heading, state.elevation)
                ##         for state in path_states]
                trajectory = [path_element_from_observation(ob) for ob in path_observations]
                this_trajs.append({
                    'instr_id': path_observations[0]['instr_id'],
                    'instr_encoding': path_observations[0]['instr_encoding'],
                    'trajectory': trajectory,
                    'observations': path_observations,
                    'actions': path_actions,
                    'score': inf_state.score,
                    'scores': path_scores,
                })
            trajs.append(this_trajs)
        return trajs

    def set_beam_size(self, beam_size):
        if self.env.beam_size < beam_size:
            self.env.set_beam_size(beam_size)
        self.beam_size = beam_size

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, beam_size=1):
        ''' Evaluate once on each instruction in the current environment '''
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        self.set_beam_size(beam_size)
        return super(Seq2SeqAgent, self).test()

    def train(self, encoder_optimizer, decoder_optimizer, n_iters, feedback='teacher'):
        ''' Train for a given number of iterations '''
        assert all(f in self.feedback_options for f in feedback.split("+"))
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        it = range(1, n_iters + 1)
        try:
            import tqdm
            it = tqdm.tqdm(it)
        except:
            pass
        for _ in it:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            self._rollout_with_loss()
            self.loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

    def _encoder_and_decoder_paths(self, base_path):
        return base_path + "_enc", base_path + "_dec"

    def save(self, path):
        ''' Snapshot models '''
        encoder_path, decoder_path = self._encoder_and_decoder_paths(path)
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, path, **kwargs):
        ''' Loads parameters (but not training state) '''
        encoder_path, decoder_path = self._encoder_and_decoder_paths(path)
        self.encoder.load_state_dict(torch.load(encoder_path, **kwargs))
        self.decoder.load_state_dict(torch.load(decoder_path, **kwargs))
