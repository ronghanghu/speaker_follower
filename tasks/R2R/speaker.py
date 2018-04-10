import json
import sys
import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils import vocab_padding_idx, vocab_bos_idx, vocab_eos_idx, flatten, try_cuda

from env import IGNORE_ACTION_INDEX, index_action_tuple

from follower import process_instruction_batch

class Seq2SeqSpeaker(object):
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, encoder, decoder, instruction_len):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

        self.encoder = encoder
        self.decoder = decoder
        self.instruction_len = instruction_len

        self.losses = []

    def write_results(self):
        with open(self.results_path, 'w') as f:
            json.dump(self.results, f)

    def test(self):
        self.env.reset_epoch()
        self.losses = []
        self.results = {}

        looped = False
        while True:
            rollout_results = self.rollout()
            for result in rollout_results:
                if result['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[result['instr_id']] = result
            if looped:
                break

    def n_inputs(self):
        return self.decoder.vocab_size

    def n_outputs(self):
        return self.decoder.vocab_size-1 # Model doesn't output start

    def _feature_variable(self, obs, beamed=False):
        ''' Extract precomputed features into variable. '''
        features = np.stack([ob['feature'] for ob in (flatten(obs) if beamed else obs)])
        return try_cuda(Variable(torch.from_numpy(features), requires_grad=False))

    def _proc_batch(self, starting_world_states):
        # note: return values will be in sorted order
        path_obs, path_actions = self.env.shortest_paths_to_goals(starting_world_states)
        seq_lengths = [len(a) for a in path_actions]
        max_path_length = max(seq_lengths)
        path_obs, path_actions, seq_lengths = zip(*sorted(zip(path_obs, path_actions, seq_lengths), key=lambda p: p[2], reverse=True))

        for o, a in zip(path_obs, path_actions):
            assert len(o) == len(a) + 1

        assert seq_lengths[0] == max_path_length

        batch_size = len(path_obs)
        assert batch_size == len(path_actions)

        batched_actions = np.full((batch_size, max_path_length), IGNORE_ACTION_INDEX, dtype='int64')
        for i, actions in enumerate(path_actions):
            batched_actions[i,0:len(actions)] = [index_action_tuple(tpl) for tpl in actions]
        batched_actions = torch.from_numpy(batched_actions).long()

        image_feature_shape = path_obs[0][0]['feature'].shape
        batched_image_features = np.zeros((batch_size, max_path_length) + image_feature_shape, dtype='float32')
        for i, obs in enumerate(path_obs):
            # don't include the last state, which should result after the stop action
            obs = obs[:-1]
            batched_image_features[i,0:len(obs)] = [ob['feature'] for ob in obs]
        batched_image_features = torch.from_numpy(batched_image_features)

        mask = (batched_actions == IGNORE_ACTION_INDEX).byte()

        start_obs = [obs[0] for obs in path_obs]

        return start_obs, \
               try_cuda(Variable(batched_image_features, requires_grad=False)), \
               try_cuda(Variable(batched_actions, requires_grad=False)), \
               try_cuda(mask), \
               list(seq_lengths)

    def rollout(self):
        starting_world_states = self.env.reset()
        start_obs, batched_image_features, batched_actions, path_mask, path_lengths = self._proc_batch(starting_world_states)
        instr_seq, instr_seq_mask, instr_seq_lengths = process_instruction_batch(start_obs, beamed=False)

        batch_size = len(start_obs)

        ctx,h_t,c_t = self.encoder(batched_actions, batched_image_features, path_lengths)

        w_t = try_cuda(Variable(torch.from_numpy(np.full((batch_size,), vocab_bos_idx, dtype='int64')).long(),
                                requires_grad=False))
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        outputs = []
        for i in range(batch_size):
            outputs.append({
                'instr_id': start_obs[i]['instr_id'],
                'word_indices': [],
            })

        # Do a sequence rollout and calculate the loss
        self.loss = 0
        sequence_scores = try_cuda(torch.zeros(batch_size))
        for t in range(self.instruction_len):
            h_t,c_t,alpha,logit = self.decoder(w_t.view(-1, 1), h_t, c_t, ctx, path_mask)
            # Supervised training

            # BOS are not part of the encoded sequences
            target = instr_seq[:,t].contiguous()

            # Determine next model inputs
            if self.feedback == 'teacher':
                w_t = target
            elif self.feedback == 'argmax':
                _,w_t = logit.max(1)        # student forcing - argmax
                w_t = w_t.detach()
            elif self.feedback == 'sample':
                probs = F.softmax(logit)    # sampling an action from model
                w_t = probs.multinomial(1).detach().squeeze(-1)
            else:
                sys.exit('Invalid feedback option')

            log_probs = F.log_softmax(logit, dim=1)
            word_scores = -F.nll_loss(log_probs, w_t, ignore_index=vocab_padding_idx, reduce=False)
            sequence_scores += word_scores.data

            if self.feedback == 'teacher':
                self.loss -= torch.mean(word_scores)
            else:
                self.loss += F.nll_loss(log_probs, target, ignore_index=vocab_padding_idx, reduce=True, size_average=True)

            for i in range(batch_size):
                word_idx = w_t[i].data[0]
                if not ended[i]:
                    outputs[i]['word_indices'].append(word_idx)
                    outputs[i]['score'] = sequence_scores[i]
                if word_idx == vocab_eos_idx:
                    ended[i] = True

            # print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, world_states[0], a_t.data[0], sequence_scores[0]))

            # Early exit if all ended
            if ended.all():
                break

        for item in outputs:
            item['words'] = self.env.tokenizer.decode_sentence(item['word_indices'], break_on_eos=True, join=False)

        self.losses.append(self.loss.data[0])
        return outputs

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
        self.beam_size = beam_size
        self.env.reset_epoch()
        self.losses = []
        self.results = {}

        # We rely on env showing the entire batch before repeating anything
        looped = False
        while True:
            rollout_results = self.rollout()
            for result in rollout_results:
                if result['instr_id'] in self.results:
                    looped = True
                else:
                    self.results[result['instr_id']] = result
            if looped:
                break

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
            self.rollout()
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
