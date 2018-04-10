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

from utils import vocab_padding_idx, flatten, structured_map, try_cuda

from env import FOLLOWER_MODEL_ACTIONS, FOLLOWER_ENV_ACTIONS, IGNORE_ACTION_INDEX, index_action_tuple

InferenceState = namedtuple("InferenceState", "prev_inference_state, world_state, observation, flat_index, last_action, action_count, score")

def backchain_inference_states(last_inference_state):
    states = []
    observations = []
    actions = []
    inf_state = last_inference_state
    while inf_state is not None:
        states.append(inf_state.world_state)
        observations.append(inf_state.observation)
        actions.append(inf_state.last_action)
        inf_state = inf_state.prev_inference_state
    return list(reversed(states)), list(reversed(observations)), list(reversed(actions))[1:] # exclude start action

def process_instruction_batch(obs, beamed=False):
    seq_tensor = np.array([ob['instr_encoding'] for ob in (flatten(obs) if beamed else obs)])
    seq_lengths = np.argmax(seq_tensor == vocab_padding_idx, axis=1)
    seq_lengths[seq_lengths == 0] = seq_tensor.shape[1] # Full length

    seq_tensor = torch.from_numpy(seq_tensor)
    seq_lengths = torch.from_numpy(seq_lengths)

    mask = (seq_tensor == vocab_padding_idx)[:, :seq_lengths[0]]

    return try_cuda(Variable(seq_tensor, requires_grad=False).long()), \
           try_cuda(mask.byte()), \
           list(seq_lengths)

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_results(self):
        with open(self.results_path, 'w') as f:
            json.dump(self.results, f)

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
            #     beam_results = self.beam_search(1, load_next_minibatch=False)
            #     assert len(rollout_results) == len(beam_results)
            #     for rollout_traj, beam_trajs in zip(rollout_results, beam_results):
            #         assert rollout_traj['instr_id'] == beam_trajs[0]['instr_id']
            #         assert rollout_traj['trajectory'] == beam_trajs[0]['trajectory']
            #         assert np.allclose(rollout_traj['score'], beam_trajs[0]['score'])
            #     print("passed check: beam_search with beam_size=1")
            #
            #     self.env.set_beam_size(10)
            #     beam_results = self.beam_search(10, load_next_minibatch=False)
            #     assert len(rollout_results) == len(beam_results)
            #     for rollout_traj, beam_trajs in zip(rollout_results, beam_results):
            #         rollout_score = rollout_traj['score']
            #         rollout_scores.append(rollout_score)
            #         beam_score = beam_trajs[0]['score']
            #         beam_10_scores.append(beam_score)
            #         # assert rollout_score <= beam_score
            #     self.env.set_beam_size(1)
            #     # print("passed check: beam_search with beam_size=10")
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
        self.steps = random.sample(list(range(-11,1)), len(obs))
        ended = [False] * len(obs)
        for t in range(30):
            actions = []
            for i,ob in enumerate(obs):
                if self.steps[i] >= 5:
                    actions.append((0, 0, 0)) # do nothing, i.e. end
                    ended[i] = True
                elif self.steps[i] < 0:
                    actions.append((0, 1, 0)) # turn right (direction choosing)
                    self.steps[i] += 1
                elif len(ob['navigableLocations']) > 1:
                    actions.append((1, 0, 0)) # go forward
                    self.steps[i] += 1
                else:
                    actions.append((0, 1, 0)) # turn right until we can go forward
            world_states = self.env.step(world_states, actions)
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
        all_obs, all_actions = self.env.shortest_paths_to_goals(world_states)
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
    model_actions = FOLLOWER_MODEL_ACTIONS
    env_actions = FOLLOWER_ENV_ACTIONS
    start_index = model_actions.index('<start>')
    ignore_index = IGNORE_ACTION_INDEX
    forward_index = model_actions.index('forward')
    end_index = model_actions.index('<end>')
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, encoder, decoder, episode_len=20, beam_size=1):
        super(Seq2SeqAgent, self).__init__(env, results_path)
        self.encoder = encoder
        self.decoder = decoder
        self.episode_len = episode_len
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.beam_size = beam_size

    @staticmethod
    def n_inputs():
        return len(Seq2SeqAgent.model_actions)

    @staticmethod
    def n_outputs():
        return len(Seq2SeqAgent.model_actions)-2 # Model doesn't output start or ignore

    def _feature_variable(self, obs, beamed=False):
        ''' Extract precomputed features into variable. '''
        features = np.stack([ob['feature'] for ob in (flatten(obs) if beamed else obs)])
        return try_cuda(Variable(torch.from_numpy(features), requires_grad=False))

    def _teacher_action(self, obs, ended):
        ''' Extract teacher actions into variable. '''
        a = torch.LongTensor(len(obs))
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            action_tuple = ob['teacher']
            index = index_action_tuple(action_tuple)
            if index == self.end_index and ended[i]:
                index = self.ignore_index
            a[i] = index
        return try_cuda(Variable(a, requires_grad=False))


    def _proc_batch(self, obs, beamed=False):
        return process_instruction_batch(obs, beamed)

    def rollout(self):
        if self.beam_size == 1:
            return self._rollout_with_loss()
        else:
            assert self.beam_size >= 1
            beams = self.beam_search(self.beam_size)
            return [beam[0] for beam in beams]

    def _rollout_with_loss(self):
        world_states = self.env.reset(sort=True)
        obs = self.env.observe(world_states)
        obs = np.array(obs)
        batch_size = len(obs)

        # get mask and lengths
        seq, seq_mask, seq_lengths = self._proc_batch(obs)

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob)],
            'actions': []
        } for ob in obs]

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        # TODO consider not feeding this into the decoder, and just using attention
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        # Initial action
        a_t = try_cuda(Variable(torch.ones(batch_size).long() * self.model_actions.index('<start>'),
                    requires_grad=False))
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        self.loss = 0
        env_action = [None] * batch_size
        sequence_scores = try_cuda(torch.zeros(batch_size))
        for t in range(self.episode_len):

            f_t = self._feature_variable(obs) # Image features from obs
            h_t,c_t,alpha,image_attention,logit = self.decoder(a_t.view(-1, 1), f_t, h_t, c_t, ctx, seq_mask)
            # Mask outputs where agent can't move forward
            for i,ob in enumerate(obs):
                if len(ob['navigableLocations']) <= 1:
                    logit[i, self.model_actions.index('forward')] = -float('inf')

            # Supervised training
            target = self._teacher_action(obs, ended)
            self.loss += self.criterion(logit, target)

            # Determine next model inputs
            if self.feedback == 'teacher':
                a_t = target                # teacher forcing
            elif self.feedback == 'argmax':
                _,a_t = logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                probs = F.softmax(logit)    # sampling an action from model
                a_t = probs.multinomial(1).detach().squeeze(-1)
            else:
                sys.exit('Invalid feedback option')

            action_scores = -F.cross_entropy(logit, a_t, ignore_index=self.ignore_index, reduce=False).data
            sequence_scores += action_scores

            # dfried: I changed this so that the ended list is updated afterward; this causes <end> to be added as the last action, along with its score, and the final world state will be duplicated (to more closely match beam search)
            # Make environment action
            for i in range(batch_size):
                action_idx = a_t[i].data[0]
                env_action[i] = self.env_actions[action_idx]

            world_states = self.env.step(world_states, env_action)
            obs = self.env.observe(world_states)
            # print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, world_states[0], a_t.data[0], sequence_scores[0]))

            # Save trajectory output
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['trajectory'].append(path_element_from_observation(ob))
                    traj[i]['score'] = sequence_scores[i]
                    traj[i]['actions'].append(a_t.data[i])

            # Update ended list
            for i in range(batch_size):
                action_idx = a_t[i].data[0]
                if action_idx == self.model_actions.index('<end>'):
                    ended[i] = True

            # Early exit if all ended
            if ended.all():
                break

        #self.losses.append(self.loss.data[0] / self.episode_len)
        # shouldn't divide by the episode length because of masking
        self.losses.append(self.loss.data[0])
        return traj

    def beam_search(self, beam_size, load_next_minibatch=True):
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
                            last_action=self.start_index,
                            action_count=0,
                            score=0.0)]
            for i, (ws, o) in enumerate(zip(world_states, obs))
        ]

        # Do a sequence rollout and calculate the loss
        for t in range(self.episode_len):
            flat_indices = []
            beam_indices = []
            a_t_list = []
            for beam_index, beam in enumerate(beams):
                for inf_state in beam:
                    beam_indices.append(beam_index)
                    flat_indices.append(inf_state.flat_index)
                    a_t_list.append(inf_state.last_action)

            a_t = try_cuda(Variable(torch.LongTensor(a_t_list), requires_grad=False))
            flat_obs = flatten(obs)
            f_t = self._feature_variable(flat_obs) # Image features from obs

            h_t,c_t,alpha,image_attention,logit = self.decoder(a_t.view(-1, 1), f_t, h_t[flat_indices], c_t[flat_indices], ctx[beam_indices], seq_mask[beam_indices])

            # Mask outputs where agent can't move forward
            no_forward_mask = [len(ob['navigableLocations']) <= 1 for ob in flat_obs]

            for i,no_forward in enumerate(no_forward_mask):
                if no_forward:
                    logit[i, self.forward_index] = -float('inf')

            log_probs = F.log_softmax(logit, dim=1).data

            # force ending if we've reached the max time steps
            # if t == self.episode_len - 1:
            #     action_scores = log_probs[:,self.end_index].unsqueeze(-1)
            #     action_indices = torch.from_numpy(np.full((log_probs.size()[0], 1), self.end_index))
            # else:
            action_scores, action_indices = log_probs.topk(min(beam_size, logit.size()[1]), dim=1)
            assert action_scores.size() == action_indices.size()

            start_index = 0
            new_beams = []
            assert len(beams) == len(world_states)
            all_successors = []
            for beam_index, (beam, beam_world_states) in enumerate(zip(beams, world_states)):
                successors = []
                end_index = start_index + len(beam)
                assert len(beam_world_states) == len(beam)
                if beam:
                    for inf_index, (inf_state, world_state, action_score_row, action_index_row) in \
                            enumerate(zip(beam, beam_world_states, action_scores[start_index:end_index], action_indices[start_index:end_index])):
                        for action_score, action_index in zip(action_score_row, action_index_row):
                            flat_index = start_index + inf_index
                            if no_forward_mask[flat_index] and action_index == self.forward_index:
                                continue
                            successors.append(
                                InferenceState(prev_inference_state=inf_state,
                                               world_state=world_state, # will be updated later after successors are pruned
                                               observation=None, # will be updated later after successors are pruned
                                               flat_index=flat_index,
                                               last_action=action_index,
                                               action_count=inf_state.action_count + 1,
                                               score=inf_state.score + action_score)
                            )
                start_index = end_index
                successors = sorted(successors, key=lambda t: t.score, reverse=True)[:beam_size]
                all_successors.append(successors)

            successor_world_states = [
                [inf_state.world_state for inf_state in successors]
                for successors in all_successors
            ]

            successor_env_actions = [
                [self.env_actions[inf_state.last_action] for inf_state in successors]
                for successors in all_successors
            ]

            successor_world_states = self.env.step(successor_world_states, successor_env_actions, beamed=True)
            successor_obs = self.env.observe(successor_world_states, beamed=True)

            all_successors = structured_map(lambda inf_state, world_state, obs: inf_state._replace(world_state=world_state, observation=obs),
                                   all_successors, successor_world_states, successor_obs, nested=True)

            # if all_successors[0]:
            #     print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, all_successors[0][0].world_state, all_successors[0][0].last_action, all_successors[0][0].score))

            for beam_index, successors in enumerate(all_successors):
                new_beam = []
                for successor in successors:
                    if successor.last_action == self.end_index or t == self.episode_len - 1:
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

        for beam_index, this_completed in enumerate(completed):
            assert this_completed
            this_trajs = []
            for inf_state in sorted(this_completed, key=lambda t: t.score, reverse=True)[:beam_size]:
                path_states, path_observations, path_actions = backchain_inference_states(inf_state)
                # this will have messed-up headings for (at least some) starting locations because of
                # discretization, so read from the observations instead
                ## path = [(obs.viewpointId, state.heading, state.elevation)
                ##         for state in path_states]
                trajectory = [path_element_from_observation(ob) for ob in path_observations]
                this_trajs.append({
                    'instr_id': path_observations[0]['instr_id'],
                    'trajectory': trajectory,
                    # 'observations': path_observations,
                    'actions': path_actions,
                    'score': inf_state.score,
                })
            trajs.append(this_trajs)
        return trajs

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
        if self.env.beam_size < beam_size:
            self.env.set_beam_size(beam_size)
        self.beam_size = beam_size
        super(Seq2SeqAgent, self).test()

    def train(self, encoder_optimizer, decoder_optimizer, n_iters, feedback='teacher'):
        ''' Train for a given number of iterations '''
        assert feedback in self.feedback_options
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

    def save(self, encoder_path, decoder_path):
        ''' Snapshot models '''
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, encoder_path, decoder_path):
        ''' Loads parameters (but not training state) '''
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))
