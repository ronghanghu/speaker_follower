
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse

import utils
from utils import read_vocab,write_vocab,build_vocab,Tokenizer,vocab_padding_idx,timeSince, try_cuda
from env import R2RBatch, ImageFeatures, FOLLOWER_ENV_ACTIONS, IGNORE_ACTION_INDEX
import model
from model import SpeakerEncoderLSTM, SpeakerDecoderLSTM
from speaker import Seq2SeqSpeaker
import eval_speaker


TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'
RESULT_DIR = 'tasks/R2R/speaker/results/'
SNAPSHOT_DIR = 'tasks/R2R/speaker/snapshots/'
PLOT_DIR = 'tasks/R2R/speaker/plots/'

# TODO: how much is this truncating instructions?
MAX_INSTRUCTION_LENGTH = 80

batch_size = 100
max_episode_len = 20
word_embedding_size = 256
action_embedding_size = 32
hidden_size = 512
bidirectional = False
dropout_ratio = 0.5
feedback_method = 'teacher' # teacher or sample
learning_rate = 0.0001
weight_decay = 0.0005
n_iters = 5000 if feedback_method == 'teacher' else 20000
FEATURE_SIZE = 2048
log_every=100
#log_every=20

def get_model_prefix(args):
    model_prefix = 'speaker_%s_imagenet' % (feedback_method)
    model_prefix = model_prefix + "_" + args.image_feature_type
    return model_prefix

def eval_model(agent, results_path, use_dropout, feedback, allow_cheat=False):
    agent.results_path = results_path
    agent.test(use_dropout=use_dropout, feedback=feedback, allow_cheat=allow_cheat)

def train(args, train_env, encoder, decoder, n_iters, log_every=log_every, val_envs=None):
    ''' Train on training set, validating on both seen and unseen. '''

    if val_envs is None:
        val_envs = {}

    agent = Seq2SeqSpeaker(train_env, "", encoder, decoder, MAX_INSTRUCTION_LENGTH)
    print('Training with %s feedback' % feedback_method)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay) 

    data_log = defaultdict(list)
    start = time.time()
   
    for idx in range(0, n_iters, log_every):

        interval = min(log_every,n_iters-idx)
        iter = idx + interval
        data_log['iteration'].append(iter)

        # Train for log_every interval
        agent.train(encoder_optimizer, decoder_optimizer, interval, feedback=feedback_method)
        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        # Run validation
        for env_name, (env, evaluator) in val_envs.items():
            agent.env = env
            # Get validation loss under the same conditions as training
            agent.test(use_dropout=True, feedback=feedback_method, allow_cheat=True)
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)

            agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, get_model_prefix(args), env_name, iter)
            # Get validation distance from goal under test evaluation conditions
            agent.test(use_dropout=False, feedback='argmax')
            agent.write_results()
            print("evaluating on {}".format(env_name))
            score_summary = evaluator.score_file(agent.results_path, verbose=True)
            print()

            loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            for metric,val in score_summary.items():
                data_log['%s %s' % (env_name,metric)].append(val)
                if metric in ['bleu', 'unpenalized_bleu']:
                    loss_str += ', %s: %.3f' % (metric, val)

        agent.env = train_env

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))

        df = pd.DataFrame(data_log)
        df.set_index('iteration')
        df_path = '%s%s_log.csv' % (PLOT_DIR, get_model_prefix(args))
        df.to_csv(df_path)
        
        split_string = "-".join(train_env.splits)
        enc_path = '%s%s_%s_enc_iter_%d' % (SNAPSHOT_DIR, get_model_prefix(args), split_string, iter)
        dec_path = '%s%s_%s_dec_iter_%d' % (SNAPSHOT_DIR, get_model_prefix(args), split_string, iter)
        agent.save(enc_path, dec_path)


def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen']), TRAINVAL_VOCAB)


def make_env_and_models(args, train_vocab_path, train_splits, test_splits):
    setup()
    image_features = ImageFeatures.from_args(args)
    vocab = read_vocab(train_vocab_path)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INSTRUCTION_LENGTH)
    train_env = R2RBatch(image_features, batch_size=batch_size, splits=train_splits, tokenizer=tok)

    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = try_cuda(SpeakerEncoderLSTM(len(FOLLOWER_ENV_ACTIONS), action_embedding_size, FEATURE_SIZE,
                                          enc_hidden_size, IGNORE_ACTION_INDEX, dropout_ratio, bidirectional=bidirectional))
    decoder = try_cuda(SpeakerDecoderLSTM(len(vocab), word_embedding_size, hidden_size, dropout_ratio))

    test_envs = {
        split: (R2RBatch(image_features, batch_size=batch_size, splits=[split], tokenizer=tok), eval_speaker.SpeakerEvaluation([split]))
        for split in test_splits
    }
    return train_env, test_envs, encoder, decoder

def train_val(args):
    ''' Train on the training set, and validate on seen and unseen splits. '''
    train_env, val_envs, encoder, decoder = make_env_and_models(args, TRAIN_VOCAB, ['train'], ['train_subset', 'val_seen', 'val_unseen'])
    train(args, train_env, encoder, decoder, n_iters, val_envs=val_envs)

def test_submission(args):
    ''' Train on combined training and validation sets, and generate test submission. '''
    train_env, test_envs, encoder, decoder = make_env_and_models(args, TRAINVAL_VOCAB, ['train', 'val_seen', 'val_unseen'], ['test'])
    test_env = test_envs['test']

    train(args, train_env, encoder, decoder, n_iters)

    agent = Seq2SeqSpeaker(test_env, "", encoder, decoder, MAX_INSTRUCTION_LENGTH)
    agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, get_model_prefix(args), 'test', 20000)
    agent.test(use_dropout=False, feedback='argmax')
    agent.write_results()


def make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_feature_type", choices=["mean_pooled", "none", "random", "attention"], default="mean_pooled")
    parser.add_argument("--image_attention_type", choices=["feedforward", "multiplicative"], default="feedforward")
    parser.add_argument("--image_attention_size", type=int)

    parser.add_argument("--mean_pooled_image_feature_store", default="img_features/ResNet-152-imagenet.tsv")
    parser.add_argument("--convolutional_image_feature_store", default="img_features/imagenet_convolutional")
    return parser

if __name__ == "__main__":
    utils.run(make_arg_parser(), train_val)
    #test_submission()

