import argparse
import utils
import train_speaker
from speaker import Seq2SeqSpeaker

def selfplay_setup(args):
    train_splits = ['train']
    pred_splits = args.pred_splits
    vocab = train_speaker.TRAIN_VOCAB

    train_env, pred_envs, encoder, decoder = train_speaker.make_env_and_models(args, vocab, train_splits, pred_splits, test_instruction_limit=1)
    agent = Seq2SeqSpeaker(train_env, "", encoder, decoder, train_speaker.MAX_INSTRUCTION_LENGTH)
    return agent, train_env, pred_envs

def entry_point(args):
    agent, train_env, val_envs = selfplay_setup(args)
    agent.load(args.model_prefix)

    for env_name, (env, evaluator) in val_envs.items():
        agent.env = env
        agent.env.print_progress = True

        ## gold
        # gold_results = agent.test(use_dropout=False, feedback='teacher', allow_cheat=True)
        # gold_score_summary = evaluator.score_results(gold_results, verbose=False)
        #
        # for metric,val in gold_score_summary.items():
        #     print("gold {} {}\t{}".format(env_name, metric, val))
        #
        # if args.gold_results_output_file:
        #     fname = "{}_{}.json".format(args.gold_results_output_file, env_name)
        #     with open(fname, 'w') as f:
        #         utils.pretty_json_dump(gold_results, f)

        ## predicted
        pred_results = agent.test(use_dropout=False, feedback='argmax')
        pred_score_summary, pred_replaced_gt = evaluator.score_results(pred_results, verbose=False)

        for metric,val in pred_score_summary.items():
            print("pred {} {}\t{}".format(env_name, metric, val))

        fname = "{}_{}.json".format(args.pred_results_output_file, env_name)
        with open(fname, 'w') as f:
            utils.pretty_json_dump(pred_replaced_gt, f)

def make_arg_parser():
    parser = train_speaker.make_arg_parser()
    parser.add_argument("model_prefix")
    parser.add_argument("pred_results_output_file")
    parser.add_argument("--pred_splits", nargs="+", default=["train_selfplay"])
    #parser.add_argument("--beam_size", type=int, default=1)
    return parser

if __name__ == "__main__":
    utils.run(make_arg_parser(), entry_point)
