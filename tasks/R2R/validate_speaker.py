import utils
import train_speaker


def validate_entry_point(args):
    agent, train_env, val_envs = train_speaker.train_setup(args)
    agent.load(args.model_prefix)

    for env_name, (val_env, evaluator) in sorted(val_envs.items()):
        agent.env = val_env

        # # gold
        # gold_results = agent.test(
        #     use_dropout=False, feedback='teacher', allow_cheat=True)
        # gold_score_summary = evaluator.score_results(
        #     gold_results, verbose=False)
        #
        # for metric,val in gold_score_summary.items():
        #     print("gold {} {}\t{}".format(env_name, metric, val))
        #
        # if args.gold_results_output_file:
        #     fname = "{}_{}.json".format(
        #         args.gold_results_output_file, env_name)
        #     with open(fname, 'w') as f:
        #         utils.pretty_json_dump(gold_results, f)

        # predicted
        pred_results = agent.test(use_dropout=False, feedback='argmax')
        pred_score_summary, _ = evaluator.score_results(
            pred_results, verbose=False)

        for metric, val in pred_score_summary.items():
            print("pred {} {}\t{}".format(env_name, metric, val))

        if args.pred_results_output_file:
            fname = "{}_{}.json".format(
                args.pred_results_output_file, env_name)
            with open(fname, 'w') as f:
                utils.pretty_json_dump(pred_results, f)


def make_arg_parser():
    parser = train_speaker.make_arg_parser()
    parser.add_argument("model_prefix")
    parser.add_argument("--gold_results_output_file")
    parser.add_argument("--pred_results_output_file")
    # parser.add_argument("--beam_size", type=int, default=1)
    return parser


if __name__ == "__main__":
    utils.run(make_arg_parser(), validate_entry_point)
