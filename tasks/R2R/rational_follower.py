import argparse
import utils
import train
import train_speaker
from follower import Seq2SeqAgent
import pprint
import json
import MatterSim
import env

import numpy as np

from collections import namedtuple, Counter

#FollowerCandidate = namedtuple("FollowerCandidate", "instr_id, observations, actions, instr_encoding, follower_score, speaker_score")

def run_rational_follower(envir, evaluator, follower, speaker, beam_size, include_gold=False, output_file=None, compute_oracle=False, mask_undo=False):
    follower.env = envir
    envir.reset_epoch()

    feedback_method = 'argmax'
    follower.encoder.eval()
    follower.decoder.eval()

    speaker.encoder.eval()
    speaker.decoder.eval()

    follower.set_beam_size(beam_size)

    candidate_lists_by_instr_id = {}

    looped = False
    while True:
        if include_gold:
            follower.feedback = 'teacher'
            gold_candidates = follower._rollout_with_loss()
        else:
            gold_candidates = []

        follower.feedback = feedback_method
        beam_candidates = follower.beam_search(beam_size, load_next_minibatch=not include_gold, mask_undo=mask_undo)

        if include_gold:
            assert len(gold_candidates) == len(beam_candidates)
            for i, bc in enumerate(beam_candidates):
                assert gold_candidates[i]['instr_id'] == bc[0]['instr_id']
                bc.insert(0, gold_candidates[i])

        cand_obs = []
        cand_actions = []
        cand_instr = []
        for candidate in utils.flatten(beam_candidates):
            cand_obs.append(candidate['observations'])
            cand_actions.append(candidate['actions'])
            cand_instr.append(candidate['instr_encoding'])

        speaker_scored_candidates, _ = speaker._score_obs_actions_and_instructions(cand_obs, cand_actions, cand_instr, feedback='teacher')
        assert len(speaker_scored_candidates) == sum(len(l) for l in beam_candidates)
        start_index = 0
        for instance_candidates in beam_candidates:
            for i, candidate in enumerate(instance_candidates):
                speaker_scored_candidate = speaker_scored_candidates[start_index + i]
                assert candidate['instr_id'] == speaker_scored_candidate['instr_id']
                candidate['follower_score'] = candidate['score']
                candidate['speaker_score'] = speaker_scored_candidate['score']
                if compute_oracle:
                    candidate['eval_result'] = evaluator._score_item(candidate['instr_id'], candidate['trajectory'])._asdict()
            start_index += len(instance_candidates)
            assert utils.all_equal([i['instr_id'] for i in instance_candidates])
            instr_id = instance_candidates[0]['instr_id']
            if instr_id in candidate_lists_by_instr_id:
                looped = True
            else:
                candidate_lists_by_instr_id[instr_id] = instance_candidates
        if looped:
            break

    follower_scores = [cand['follower_score']
                       for lst in candidate_lists_by_instr_id.values()
                       for cand in lst]
    speaker_scores = [cand['speaker_score']
                       for lst in candidate_lists_by_instr_id.values()
                       for cand in lst]

    speaker_std = np.std(speaker_scores)
    follower_std = np.std(follower_scores)

    accuracies_by_weight = {}
    index_counts_by_weight = {}

    for speaker_weight in [0.95]:  # Use 0.95 weight
    # for speaker_weight in np.arange(0, 20 + 1) / 20.0:
        results = {}
        index_count = Counter()

        speaker_scaled_weight = speaker_weight / speaker_std
        follower_scaled_weight = (1 - speaker_weight) / follower_std

        for instr_id, candidates in candidate_lists_by_instr_id.items():
            best_ix, best_cand = max(enumerate(candidates), key=lambda tp: tp[1]['speaker_score'] * speaker_scaled_weight + tp[1]['follower_score'] * follower_scaled_weight)
            results[instr_id] = best_cand
            index_count[best_ix] += 1

        score_summary, _ = evaluator.score_results(results)

        accuracies_by_weight[speaker_weight] = score_summary
        index_counts_by_weight[speaker_weight] = index_count

    if compute_oracle:
        oracle_results = {}
        oracle_index_count = Counter()
        for instr_id, candidates in candidate_lists_by_instr_id.items():
            best_ix, best_cand = min(enumerate(candidates), key=lambda tp: tp[1]['eval_result']['nav_error'])
            # if include_gold and not best_cand['eval_result']['success']:
            #     print("--compute_oracle and --include_gold but not success!")
            #     print(best_cand)
            oracle_results[instr_id] = best_cand
            oracle_index_count[best_ix] += 1

        oracle_score_summary, _ = evaluator.score_results(oracle_results)
        print("oracle results:")
        pprint.pprint(oracle_score_summary)
        pprint.pprint(sorted(oracle_index_count.items()))

    if output_file:
        with open(output_file, 'w') as f:
            for candidate_list in candidate_lists_by_instr_id.values():
                for i, candidate in enumerate(candidate_list):
                    del candidate['observations']
                    candidate['actions'] = [env.FOLLOWER_MODEL_ACTIONS[ac] for ac in candidate['actions']]
                    candidate['scored_actions'] = list(zip(candidate['actions'], candidate['scores']))
                    candidate['instruction'] = envir.tokenizer.decode_sentence(candidate['instr_encoding'], break_on_eos=False, join=True)
                    del candidate['instr_encoding']
                    del candidate['trajectory']
                    candidate['rank'] = i
                    candidate['gold'] = (include_gold and i == 0)
            utils.pretty_json_dump(candidate_lists_by_instr_id, f)

    return accuracies_by_weight, index_counts_by_weight

def validate_entry_point(args):
    follower, follower_train_env, follower_val_envs = train.train_setup(args, args.batch_size)
    load_args = {}
    if args.no_cuda:
        load_args['map_location'] = 'cpu'
    follower.load(args.follower_prefix, **load_args)

    speaker, speaker_train_env, speaker_val_envs = train_speaker.train_setup(args)
    speaker.load(args.speaker_prefix, **load_args)

    for env_name, (env, evaluator) in sorted(follower_val_envs.items()):
        if args.output_file:
            output_file = "{}_{}.json".format(args.output_file, env_name)
        else:
            output_file = None
        accuracies_by_weight, index_counts_by_weight = run_rational_follower(env, evaluator, follower, speaker, args.beam_size, include_gold=args.include_gold, output_file=output_file, compute_oracle=args.compute_oracle, mask_undo=args.mask_undo)
        pprint.pprint(accuracies_by_weight)
        pprint.pprint({w:sorted(d.items()) for w, d in index_counts_by_weight.items()})
        weight, score_summary = max(accuracies_by_weight.items(), key=lambda pair: pair[1]['success_rate'])
        print("max success_rate with weight: {}".format(weight))
        for metric,val in score_summary.items():
            print("{} {}\t{}".format(env_name, metric, val))

def make_arg_parser():
    parser = train.make_arg_parser()
    parser.add_argument("follower_prefix")
    parser.add_argument("speaker_prefix")
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--include_gold", action='store_true')
    parser.add_argument("--output_file")
    parser.add_argument("--compute_oracle", action='store_true')
    parser.add_argument("--mask_undo", action='store_true')
    return parser

if __name__ == "__main__":
    utils.run(make_arg_parser(), validate_entry_point)
