import utils
import train
import train_speaker
import pprint
from collections import Counter
import numpy as np


def generate_and_score_candidates(envir, speaker, follower, n_candidates,
                                  include_gold=False):
    follower.env = envir
    speaker.env = envir
    envir.reset_epoch()

    feedback_method = 'argmax'
    speaker.feedback = feedback_method
    follower.encoder.eval()
    follower.decoder.eval()

    speaker.encoder.eval()
    speaker.decoder.eval()

    follower.set_beam_size(1)

    candidate_lists_by_instr_id = {}

    num_candidates_per_instance = []
    looped = False
    batch_idx = 0
    while True:
        print('loaded batch %d' % batch_idx)
        batch_idx += 1
        path_obs, path_actions, gold_encoded_instructions = \
            speaker.env.gold_obs_actions_and_instructions(
                speaker.max_episode_len, load_next_minibatch=True)
        if include_gold:
            gold_candidates, gold_loss = \
                speaker._score_obs_actions_and_instructions(
                    path_obs, path_actions, gold_encoded_instructions,
                    'teacher')
        else:
            gold_candidates = []

        beam_candidates = speaker.beam_search(
            n_candidates, path_obs, path_actions)

        if include_gold:
            assert len(gold_candidates) == len(beam_candidates)
            for i, bc in enumerate(beam_candidates):
                assert gold_candidates[i]['instr_id'] == bc[0]['instr_id']
                bc.insert(0, gold_candidates[i])

        cand_words = []
        cand_word_indices = []
        cand_obs = []
        cand_actions = []
        for beam_index, this_beam in enumerate(beam_candidates):
            num_candidates_per_instance.append(len(this_beam))
            for candidate in this_beam:
                cand_obs.append(path_obs[beam_index])
                cand_actions.append(path_actions[beam_index])
                cand_words.append(candidate['words'])
                indices = candidate['word_indices']
                if indices[-1] == utils.vocab_eos_idx:
                    indices = indices[:-1]
                cand_word_indices.append(indices)

        follower_scored_candidates, _ = \
            follower._score_obs_actions_and_instructions(
                cand_obs, cand_actions, cand_word_indices)
        assert (len(follower_scored_candidates) ==
                sum(len(l) for l in beam_candidates))
        start_index = 0
        for instance_candidates in beam_candidates:
            for i, candidate in enumerate(instance_candidates):
                follower_scored_candidate = \
                    follower_scored_candidates[start_index + i]
                assert (candidate['instr_id'] ==
                        follower_scored_candidate['instr_id'])
                candidate['speaker_score'] = candidate['score']
                candidate['follower_score'] = \
                    follower_scored_candidate['score']
                candidate['actions'] = follower_scored_candidate['actions']
                if 'attentions' in candidate:
                    candidate['attentions'] = [
                        list(tens) for tens in candidate['attentions']]
                assert np.allclose(
                    np.sum(follower_scored_candidate['scores']),
                    follower_scored_candidate['score'])
            start_index += len(instance_candidates)
            assert utils.all_equal(
                [i['instr_id'] for i in instance_candidates])
            instr_id = instance_candidates[0]['instr_id']
            if instr_id in candidate_lists_by_instr_id:
                looped = True
            else:
                candidate_lists_by_instr_id[instr_id] = instance_candidates
        if looped:
            break

    print("average distinct candidates per instance: {}".format(
        np.mean(num_candidates_per_instance)))

    return candidate_lists_by_instr_id


def predict_from_candidates(candidate_lists_by_instr_id, speaker_weights):
    speaker_scores = [cand['speaker_score']
                      for lst in candidate_lists_by_instr_id.values()
                      for cand in lst]
    follower_scores = [cand['follower_score']
                       for lst in candidate_lists_by_instr_id.values()
                       for cand in lst]

    speaker_std = np.std(speaker_scores)
    follower_std = np.std(follower_scores)

    results_by_weight = {}

    for speaker_weight in np.arange(0, 20 + 1) / 20.0:
        results = {}
        index_count = Counter()

        speaker_scaled_weight = speaker_weight / speaker_std
        follower_scaled_weight = (1 - speaker_weight) / follower_std

        for instr_id, candidates in candidate_lists_by_instr_id.items():
            best_ix, best_cand = max(
                enumerate(candidates), key=lambda tp: (
                    tp[1]['speaker_score'] * speaker_scaled_weight +
                    tp[1]['follower_score'] * follower_scaled_weight))
            results[instr_id] = best_cand
            index_count[best_ix] += 1

        results_by_weight[speaker_weight] = results

    return results_by_weight


def run_rational_speaker(envir, speaker_evaluator, speaker, follower,
                         n_candidates, include_gold=False, output_file=None):
    candidate_lists_by_instr_id = generate_and_score_candidates(
        envir, speaker, follower, n_candidates)

    speaker_weights = np.arange(0, 20 + 1) / 20.0
    results_by_weight = predict_from_candidates(
        candidate_lists_by_instr_id, speaker_weights)

    scores_by_weight = {}
    for speaker_weight, results in results_by_weight.items():
        score_summary, _ = speaker_evaluator.score_results(results)
        scores_by_weight[speaker_weight] = score_summary

    if output_file:
        with open(output_file, 'w') as f:
            for candidate_list in candidate_lists_by_instr_id.values():
                for i, candidate in enumerate(candidate_list):
                    candidate['actions'] = [ac for ac in candidate['actions']]
                    candidate['scored_word_indices'] = list(
                        zip(candidate['scores'], candidate['word_indices']))
                    candidate['rank'] = i
                    candidate['gold'] = (include_gold and i == 0)
            utils.pretty_json_dump(candidate_lists_by_instr_id, f)

    return scores_by_weight, results_by_weight


def validate_entry_point(args):
    follower, follower_train_env, follower_val_envs = \
        train.train_setup(args, args.batch_size)
    load_args = {}
    if args.no_cuda:
        load_args['map_location'] = 'cpu'
    follower.load(args.follower_prefix, **load_args)

    speaker, speaker_train_env, speaker_val_envs = \
        train_speaker.train_setup(args)
    speaker.load(args.speaker_prefix, **load_args)

    for env_name, (val_env, evaluator) in sorted(speaker_val_envs.items()):
        if args.output_file:
            output_file = "{}_{}.json".format(args.output_file, env_name)
        else:
            output_file = None
        scores_by_weight, _ = run_rational_speaker(
            val_env, evaluator, speaker, follower, args.beam_size,
            include_gold=args.include_gold, output_file=output_file)
        pprint.pprint(scores_by_weight)
        weight, score_summary = max(
            scores_by_weight.items(), key=lambda pair: pair[1]['bleu'])
        print("max success_rate with weight: {}".format(weight))
        for metric, val in score_summary.items():
            print("{} {}\t{}".format(env_name, metric, val))


def make_arg_parser():
    parser = train.make_arg_parser()
    parser.add_argument("speaker_prefix")
    parser.add_argument("follower_prefix")
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--include_gold", action='store_true')
    parser.add_argument("--output_file")
    parser.add_argument("--mask_undo", action='store_true')
    return parser


if __name__ == "__main__":
    utils.run(make_arg_parser(), validate_entry_point)
