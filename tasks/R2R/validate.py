import argparse
import utils
import train
from agent import Seq2SeqAgent

def validate_entry_point(args):
    train_env, val_envs, encoder, decoder = train.make_env_and_models(args, train.TRAIN_VOCAB, ['train'], ['train_subset', 'val_seen', 'val_unseen'])
    agent = Seq2SeqAgent(train_env, "", encoder, decoder, train.max_episode_len)
    encoder_path = "{}_enc_iter_{}".format(args.model_prefix, args.iteration)
    decoder_path = "{}_dec_iter_{}".format(args.model_prefix, args.iteration)
    agent.load(encoder_path, decoder_path)

    old_env = agent.env
    for env_name, (env, evaluator) in val_envs.items():
        agent.env = env
        agent.test(use_dropout=False, feedback='argmax', beam_size=args.beam_size)
        score_summary, _ = evaluator.score_results(agent.results)

        # TODO: testing code, remove
        # score_summary_direct, _ = evaluator.score_results(agent.results)
        # assert score_summary == score_summary_direct

        for metric,val in score_summary.items():
            print("{} {}\t{}".format(env_name, metric, val))

    agent.env = old_env

def make_arg_parser():
    parser = train.make_arg_parser()
    parser.add_argument("model_prefix")
    parser.add_argument("iteration", type=int)
    parser.add_argument("--beam_size", type=int, default=1)
    return parser

if __name__ == "__main__":
    utils.run(make_arg_parser(), validate_entry_point)
