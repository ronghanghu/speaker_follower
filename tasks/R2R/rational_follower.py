import argparse
import utils
import train
import train_speaker
from follower import Seq2SeqAgent
import pprint

import numpy as np

from collections import namedtuple

#FollowerCandidate = namedtuple("FollowerCandidate", "instr_id, observations, actions, instr_encoding, follower_score, speaker_score")

def run_rational_follower(env, evaluator, follower, speaker, beam_size):
    follower.env = env
    env.reset_epoch()

    follower.feedback = 'argmax'
    follower.encoder.eval()
    follower.decoder.eval()

    speaker.encoder.eval()
    speaker.decoder.eval()

    follower.set_beam_size(beam_size)

    candidate_lists_by_instr_id = {}

    looped = False
    while True:
        beam_candidates = follower.beam_search(beam_size)

        cand_obs = []
        cand_actions = []
        cand_instr = []
        for candidate in utils.flatten(beam_candidates):
            cand_obs.append(candidate['observations'])
            cand_actions.append(candidate['actions'])
            cand_instr.append(candidate['instr_encoding'])

        speaker_scored_candidates, _ = speaker._rollout_with_instructions_obs_and_actions(cand_obs, cand_actions, cand_instr, feedback='teacher')
        assert len(speaker_scored_candidates) == sum(len(l) for l in beam_candidates)
        start_index = 0
        for instance_candidates in beam_candidates:
            for i, candidate in enumerate(instance_candidates):
                speaker_scored_candidate = speaker_scored_candidates[start_index + i]
                assert candidate['instr_id'] == speaker_scored_candidate['instr_id']
                candidate['follower_score'] = candidate['score']
                candidate['speaker_score'] = speaker_scored_candidate['score']
            start_index += len(instance_candidates)
            assert utils.all_equal([i['instr_id'] for i in instance_candidates])
            instr_id = instance_candidates[0]['instr_id']
            if instr_id in candidate_lists_by_instr_id:
                looped = True
            else:
                candidate_lists_by_instr_id[instr_id] = instance_candidates
        if looped:
            break

    accuracies_by_weight = {}

    for speaker_weight in np.arange(0, 20 + 1) / 20.0:
        results = {
            instr_id: max(candidates, key=lambda cand: cand['speaker_score'] * speaker_weight + cand['follower_score'] * (1 - speaker_weight))
            for instr_id, candidates in candidate_lists_by_instr_id.items()
        }

        score_summary, _ = evaluator.score_results(results)

        accuracies_by_weight[speaker_weight] = score_summary
    return accuracies_by_weight



def validate_entry_point(args):
    follower, follower_train_env, follower_val_envs = train.train_setup(args)
    follower.load(args.follower_prefix)

    speaker, speaker_train_env, speaker_val_envs = train_speaker.train_setup(args)

    for env_name, (env, evaluator) in follower_val_envs.items():
        accuracies_by_weight = run_rational_follower(env, evaluator, follower, speaker, args.beam_size)
        pprint.pprint(accuracies_by_weight)
        weight, score_summary = max(accuracies_by_weight.items(), key=lambda pair: pair[1]['success_rate'])
        print("max success_rate with weight: {}".format(weight))
        for metric,val in score_summary.items():
            print("{} {}\t{}".format(env_name, metric, val))

def make_arg_parser():
    parser = train.make_arg_parser()
    parser.add_argument("follower_prefix")
    parser.add_argument("speaker_prefix")
    parser.add_argument("--beam_size", type=int, default=10)
    return parser

if __name__ == "__main__":
    utils.run(make_arg_parser(), validate_entry_point)
