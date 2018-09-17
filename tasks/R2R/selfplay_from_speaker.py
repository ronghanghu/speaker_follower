import utils
import train
import train_speaker
from speaker import Seq2SeqSpeaker
from follower import Seq2SeqAgent
import rational_speaker


def selfplay_speaker_setup(args):
    train_splits = ['train']
    pred_splits = args.pred_splits
    vocab = train_speaker.TRAIN_VOCAB

    train_env, pred_envs, encoder, decoder = train_speaker.make_env_and_models(
        args, vocab, train_splits, pred_splits, test_instruction_limit=1)
    agent = Seq2SeqSpeaker(
        train_env, "", encoder, decoder, train_speaker.MAX_INSTRUCTION_LENGTH)
    return agent, train_env, pred_envs


def selfplay_follower_setup(args):
    train_splits = ['train']
    pred_splits = args.pred_splits
    vocab = train.TRAIN_VOCAB

    train_env, pred_envs, encoder, decoder = train.make_env_and_models(
        args, vocab, train_splits, pred_splits, batch_size=args.batch_size
    )
    agent = Seq2SeqAgent(
        train_env, "", encoder, decoder, train.max_episode_len,
        max_instruction_length=train.MAX_INPUT_LENGTH)
    return agent, train_env, pred_envs


def entry_point(args):
    speaker, train_env, val_envs = selfplay_speaker_setup(args)
    speaker.load(args.speaker_model_prefix)

    assert ((args.rational_speaker_weights is None) ==
            (args.follower_model_prefix is None)), \
        "must pass both --rational_speaker_weight and " \
        "--follower_model_prefix, or neither"

    pragmatic_speaker = (args.follower_model_prefix is not None)
    if pragmatic_speaker:
        follower, train_env_follower, val_envs_followers = \
            selfplay_follower_setup(args)
        follower.load(args.follower_model_prefix)
    else:
        follower = None

    for env_name, (val_env, evaluator) in sorted(val_envs.items()):
        speaker.env = val_env
        speaker.env.print_progress = True

        # predicted
        if pragmatic_speaker:
            candidate_lists_by_instr_id = \
                rational_speaker.generate_and_score_candidates(
                    val_env, speaker, follower,
                    args.rational_speaker_n_candidates, include_gold=False)
            results_by_weight = rational_speaker.predict_from_candidates(
                candidate_lists_by_instr_id, args.rational_speaker_weights)
            results_by_name = {
                'rational_speaker_{}'.format(speaker_weight): results
                for speaker_weight, results in results_by_weight.items()
            }
        else:
            pred_results = speaker.test(use_dropout=False, feedback='argmax')
            results_by_name = {'literal_speaker': pred_results}

        for name, pred_results in results_by_name.items():
            pred_score_summary, pred_replaced_gt = evaluator.score_results(
                pred_results, verbose=False)

            for metric, val in pred_score_summary.items():
                print("pred {} {} {}\t{}".format(name, env_name, metric, val))

            fname = "{}_{}_{}.json".format(
                args.pred_results_output_file, name, env_name)
            with open(fname, 'w') as f:
                utils.pretty_json_dump(pred_replaced_gt, f)


def make_arg_parser():
    # parser = train_speaker.make_arg_parser()
    # TODO: hack, this only works because the follower has extra parameters
    # that the speaker lacks!
    parser = train.make_arg_parser()
    parser.add_argument("speaker_model_prefix")
    parser.add_argument("pred_results_output_file")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--pred_splits", nargs="+",
                        default=["data_augmentation_paths"])

    # for rational self-play generation
    parser.add_argument("--follower_model_prefix",
                        help="generate data from a rational speaker "
                        "(must also pass --rational_speaker_weights")
    parser.add_argument("--rational_speaker_weights", type=float, nargs="+",
                        help="list of speaker weights in range [0.0, 1.0] to "
                        "use with rational speaker (must also pass "
                        "follower_model_prefix)")
    parser.add_argument(
        "--rational_speaker_n_candidates", type=int, default=40)
    return parser


if __name__ == "__main__":
    utils.run(make_arg_parser(), entry_point)
