
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import os
import os.path
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse

import utils
from utils import read_vocab,write_vocab,build_vocab,Tokenizer,vocab_pad_idx,timeSince, try_cuda
from env import R2RBatch, ImageFeatures
import model
from model import EncoderLSTM, AttnDecoderLSTM
from follower import Seq2SeqAgent
import eval

from vocab import SUBTRAIN_VOCAB, TRAINVAL_VOCAB, TRAIN_VOCAB

RESULT_DIR = 'tasks/R2R/results/'
SNAPSHOT_DIR = 'tasks/R2R/snapshots/'
PLOT_DIR = 'tasks/R2R/plots/'

# TODO: how much is this truncating instructions?
MAX_INPUT_LENGTH = 80

BATCH_SIZE = 100
max_episode_len = 20
word_embedding_size = 256
action_embedding_size = 32
hidden_size = 512
dropout_ratio = 0.5
#feedback_method = 'sample' # teacher or sample
learning_rate = 0.0001
weight_decay = 0.0005
FEATURE_SIZE = 2048
log_every=100
#log_every=20
save_every=1000

def get_model_prefix(args):
    image_feature_name = ImageFeatures.get_name(args)
    model_prefix = 'follower_{}_{}'.format(args.feedback_method, image_feature_name)
    if args.use_train_subset:
        model_prefix = 'trainsub_' + model_prefix
    if args.bidirectional:
        model_prefix = model_prefix + "_bidirectional"
    model_prefix = model_prefix + "_" + args.image_feature_type
    return model_prefix

def eval_model(agent, results_path, use_dropout, feedback, allow_cheat=False):
    agent.results_path = results_path
    agent.test(use_dropout=use_dropout, feedback=feedback, allow_cheat=allow_cheat)

def train(args, train_env, agent, log_every=log_every, val_envs=None):
    ''' Train on training set, validating on both seen and unseen. '''

    if val_envs is None:
        val_envs = {}

    print('Training with %s feedback' % args.feedback_method)
    encoder_optimizer = optim.Adam(agent.encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(agent.decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    data_log = defaultdict(list)
    start = time.time()

    split_string = "-".join(train_env.splits)
    def make_path(iter):
        return os.path.join(SNAPSHOT_DIR, '%s_%s_iter_%d' % (get_model_prefix(args), split_string, iter))

    best_metrics = {}
    last_model_saved = {}
    for idx in range(0, args.n_iters, log_every):
        agent.env = train_env

        interval = min(log_every,args.n_iters-idx)
        iter = idx + interval
        data_log['iteration'].append(iter)

        # Train for log_every interval
        agent.train(encoder_optimizer, decoder_optimizer, interval, feedback=args.feedback_method)
        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        save_log = []
        # Run validation
        for env_name, (env, evaluator) in val_envs.items():
            agent.env = env
            # Get validation loss under the same conditions as training
            agent.test(use_dropout=True, feedback=args.feedback_method, allow_cheat=True)
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)

            agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, get_model_prefix(args), env_name, iter)
            # Get validation distance from goal under test evaluation conditions
            agent.test(use_dropout=False, feedback='argmax')
            agent.write_results()
            score_summary, _ = evaluator.score_file(agent.results_path)

            # # TODO: testing code, remove
            # score_summary_direct, _ = evaluator.score_results(agent.results)
            # assert score_summary == score_summary_direct

            loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            for metric,val in score_summary.items():
                data_log['%s %s' % (env_name,metric)].append(val)
                if metric in ['success_rate']:
                    loss_str += ', %s: %.3f' % (metric, val)

                    key = (env_name, metric)
                    if key not in best_metrics or best_metrics[key] < val:
                        best_metrics[key] = val
                        model_path = make_path(iter) + "_%s-%s=%.3f" % (env_name, metric, val)
                        save_log.append("new best, saved model to %s" % model_path)
                        agent.save(model_path)
                        if key in last_model_saved:
                            for old_model_path in agent._encoder_and_decoder_paths(last_model_saved[key]):
                                os.remove(old_model_path)
                        last_model_saved[key] = model_path

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.n_iters),
                                             iter, float(iter)/args.n_iters*100, loss_str)))
        for s in save_log:
            print(s)

        if save_every and iter % save_every == 0:
            agent.save(make_path(iter))

        df = pd.DataFrame(data_log)
        df.set_index('iteration')
        df_path = '%s%s_log.csv' % (PLOT_DIR, get_model_prefix(args))
        df.to_csv(df_path)

def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(SUBTRAIN_VOCAB):
        write_vocab(build_vocab(splits=['sub_train']), SUBTRAIN_VOCAB)
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def make_image_attention_layer(args, image_features):
    if args.image_feature_type != 'attention':
        return None
    image_attention_size = args.image_attention_size or hidden_size
    if args.image_attention_type == 'feedforward':
        return model.FeedforwardImageAttention(image_features.feature_dim, hidden_size, image_attention_size)
    elif args.image_attention_type == 'multiplicative':
        return model.MultiplicativeImageAttention(image_features.feature_dim, hidden_size, image_attention_size)


def make_env_and_models(args, train_vocab_path, train_splits, test_splits, batch_size=BATCH_SIZE):
    setup()
    image_features = ImageFeatures.from_args(args)
    vocab = read_vocab(train_vocab_path)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = R2RBatch(image_features, batch_size=batch_size, splits=train_splits, tokenizer=tok)

    enc_hidden_size = hidden_size//2 if args.bidirectional else hidden_size
    encoder = try_cuda(EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, vocab_pad_idx,
                                   dropout_ratio, bidirectional=args.bidirectional))
    decoder = try_cuda(AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                                       action_embedding_size, hidden_size, dropout_ratio,
                                       feature_size=image_features.feature_dim,
                                       ablate_image_features=image_features.image_feature_type == "none",
                                       image_attention_layer=make_image_attention_layer(args, image_features)))
    test_envs = {split: (R2RBatch(image_features, batch_size=batch_size, splits=[split], tokenizer=tok), eval.Evaluation([split]))
                for split in test_splits}
    return train_env, test_envs, encoder, decoder

def train_setup(args, batch_size=BATCH_SIZE):
    train_splits = ['train']
    val_splits = ['train_subset', 'val_seen', 'val_unseen']

    vocab = TRAIN_VOCAB

    if args.use_train_subset:
        train_splits = ['sub_' + split for split in train_splits]
        val_splits = ['sub_' + split for split in val_splits]
        vocab = SUBTRAIN_VOCAB

    train_env, val_envs, encoder, decoder = make_env_and_models(args, vocab, train_splits, val_splits, batch_size=batch_size)
    agent = Seq2SeqAgent(train_env, "", encoder, decoder, max_episode_len)
    return agent, train_env, val_envs

def test_setup(args, batch_size=BATCH_SIZE):
    train_env, test_envs, encoder, decoder = make_env_and_models(args, TRAINVAL_VOCAB, ['train', 'val_seen', 'val_unseen'], ['test'])
    agent = Seq2SeqAgent(None, "", encoder, decoder, max_episode_len)
    return agent, train_env, test_envs

def train_val(args):
    ''' Train on the training set, and validate on seen and unseen splits. '''
    agent, train_env, val_envs = train_setup(args)
    train(args, train_env, agent, val_envs=val_envs)

def test_submission(args):
    ''' Train on combined training and validation sets, and generate test submission. '''
    agent, train_env, test_envs = test_setup(args)
    train(args, train_env, agent)

    test_env = test_envs['test']
    agent.env = test_env

    agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, get_model_prefix(args), 'test', args.n_iters)
    agent.test(use_dropout=False, feedback='argmax')
    agent.write_results()


def make_arg_parser():
    parser = argparse.ArgumentParser()
    ImageFeatures.add_args(parser)
    parser.add_argument("--feedback_method", choices=["sample", "teacher", "teacher+sample"], default="sample")
    parser.add_argument("--bidirectional", action='store_true')
    parser.add_argument("--n_iters", type=int, default=20000)

    parser.add_argument("--use_train_subset", action='store_true', help="use a subset of the original train data as val_seen and val_unseen")
    return parser

if __name__ == "__main__":
    utils.run(make_arg_parser(), train_val)
    #test_submission()

