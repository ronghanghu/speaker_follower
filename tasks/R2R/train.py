
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
from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince, try_cuda
from env import R2RBatch, ImageFeatures
import model
from model import EncoderLSTM, AttnDecoderLSTM
from agent import Seq2SeqAgent
import eval


TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'
RESULT_DIR = 'tasks/R2R/results/'
SNAPSHOT_DIR = 'tasks/R2R/snapshots/'
PLOT_DIR = 'tasks/R2R/plots/'

MAX_INPUT_LENGTH = 80

batch_size = 100
max_episode_len = 20
word_embedding_size = 256
action_embedding_size = 32
hidden_size = 512
bidirectional = False
dropout_ratio = 0.5
feedback_method = 'sample' # teacher or sample
learning_rate = 0.0001
weight_decay = 0.0005
n_iters = 5000 if feedback_method == 'teacher' else 20000
FEATURE_SIZE = 2048
log_every=100
#log_every=20

def get_model_prefix(args):
    model_prefix = 'seq2seq_%s_imagenet' % (feedback_method)
    model_prefix = model_prefix + "_" + args.image_feature_type
    return model_prefix


def train(args, train_env, encoder, decoder, n_iters, log_every=log_every, val_envs={}):
    ''' Train on training set, validating on both seen and unseen. '''

    agent = Seq2SeqAgent(train_env, "", encoder, decoder, max_episode_len)
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
            agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, get_model_prefix(args), env_name, iter)
            # Get validation loss under the same conditions as training
            agent.test(use_dropout=True, feedback=feedback_method, allow_cheat=True)
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)

            # Get validation distance from goal under test evaluation conditions
            agent.test(use_dropout=False, feedback='argmax')
            agent.write_results()
            score_summary, _ = evaluator.score(agent.results_path)
            loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            for metric,val in score_summary.items():
                data_log['%s %s' % (env_name,metric)].append(val)
                if metric in ['success_rate']:
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


def test_submission(args):
    ''' Train on combined training and validation sets, and generate test submission. '''
  
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAINVAL_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    image_features = ImageFeatures.from_args(args)
    train_env = R2RBatch(image_features, batch_size=batch_size, splits=['train', 'val_seen', 'val_unseen'], tokenizer=tok)
    
    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = try_cuda(EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                  dropout_ratio, bidirectional=bidirectional))
    decoder = try_cuda(AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                              action_embedding_size, hidden_size, dropout_ratio,
                              ablate_image_features=args.image_feature_type == "none",
                              image_attention_layer=make_image_attention_layer(args)))
    train(args, train_env, encoder, decoder, n_iters)

    # Generate test submission
    test_env = R2RBatch(image_features, batch_size=batch_size, splits=['test'], tokenizer=tok)
    agent = Seq2SeqAgent(test_env, "", encoder, decoder, max_episode_len)
    agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, get_model_prefix(args), 'test', 20000)
    agent.test(use_dropout=False, feedback='argmax')
    agent.write_results()


def train_val(args):
    ''' Train on the training set, and validate on seen and unseen splits. '''
  
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    image_features = ImageFeatures.from_args(args)
    train_env = R2RBatch(image_features, batch_size=batch_size, splits=['train'], tokenizer=tok)

    # Creat validation environments
    val_envs = {split: (R2RBatch(image_features, batch_size=batch_size, splits=[split], tokenizer=tok), eval.Evaluation([split]))
                for split in ['val_seen', 'val_unseen']}

    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = try_cuda(EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                  dropout_ratio, bidirectional=bidirectional))

    decoder = try_cuda(AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(), action_embedding_size, hidden_size,
                              dropout_ratio, ablate_image_features=args.image_feature_type == "none",
                              image_attention_layer=make_image_attention_layer(args)))
    train(args, train_env, encoder, decoder, n_iters, val_envs=val_envs)

def make_image_attention_layer(args):
    if args.image_feature_type != 'attention':
        return None
    image_attention_size = args.image_attention_size or hidden_size
    if args.image_attention_type == 'feedforward':
        return model.FeedforwardImageAttention(FEATURE_SIZE, hidden_size, image_attention_size)
    elif args.image_attention_type == 'multiplicative':
        return model.MultiplicativeImageAttention(FEATURE_SIZE, hidden_size, image_attention_size)

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

