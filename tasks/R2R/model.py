
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import try_cuda

from env import ConvolutionalImageFeatures, BottomUpImageFeatures


def make_image_attention_layers(args, image_features_list, hidden_size):
    image_attention_size = args.image_attention_size or hidden_size
    attention_mechs = []
    for featurizer in image_features_list:
        if isinstance(featurizer, ConvolutionalImageFeatures):
            if args.image_attention_type == 'feedforward':
                attention_mechs.append(MultiplicativeImageAttention(hidden_size, image_attention_size, image_feature_size=featurizer.feature_dim))
            elif args.image_attention_type == 'multiplicative':
                attention_mechs.append(FeedforwardImageAttention(hidden_size, image_attention_size, image_feature_size=featurizer.feature_dim))
        elif isinstance(featurizer, BottomUpImageFeatures):
            attention_mechs.append(BottomUpImageAttention(
                hidden_size,
                args.bottom_up_detection_embedding_size,
                args.bottom_up_detection_embedding_size,
                image_attention_size,
                featurizer.num_objects,
                featurizer.num_attributes,
                featurizer.feature_dim
            ))
        else:
            attention_mechs.append(None)
    attention_mechs = [try_cuda(mech) if mech else mech for mech in attention_mechs]
    return attention_mechs


# TODO: try variational dropout (or zoneout?)
class EncoderLSTM(nn.Module):
    ''' Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. '''

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, bidirectional=False, num_layers=1):
        super(EncoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
            hidden_size * self.num_directions
        )

    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return try_cuda(h0), try_cuda(c0)

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = inputs.size(0)
        embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(batch_size)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))

        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        return ctx,decoder_init,c_t  # (batch, seq_len, hidden_size*num_directions)
                                 # (batch, hidden_size)


class SoftDotAttention(nn.Module):
    '''Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    '''

    def __init__(self, dim):
        '''Initialize layer.'''
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None):
        '''Propagate h through the network.

        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            # -Inf masking prior to the softmax
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, h), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class FeedforwardImageAttention(nn.Module):
    def __init__(self, context_size, hidden_size, image_feature_size=2048):
        super(FeedforwardImageAttention, self).__init__()
        self.feature_size = image_feature_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.fc1_feature = nn.Conv2d(image_feature_size, hidden_size, kernel_size=1, bias=False)
        self.fc1_context = nn.Linear(context_size, hidden_size, bias=True)
        self.fc2 = nn.Conv2d(hidden_size, 1, kernel_size=1, bias=True)

    def forward(self, feature, context):
        batch_size = feature.shape[0]
        feature_hidden = self.fc1_feature(feature)
        context_hidden = self.fc1_context(context)
        context_hidden = context_hidden.unsqueeze(-1).unsqueeze(-1)
        x = feature_hidden + context_hidden
        x = self.fc2(F.relu(x))
        attention = F.softmax(x.view(batch_size, -1), 1).unsqueeze(-1) # batch_size x (width * height) x 1
        reshaped_features = feature.view(batch_size, self.feature_size, -1) # batch_size x feature_size x (width * height)
        x = torch.bmm(reshaped_features, attention) # batch_size x
        return x.squeeze(-1), attention.squeeze(-1)

class MultiplicativeImageAttention(nn.Module):
    def __init__(self, context_size, hidden_size, image_feature_size=2048):
        super(MultiplicativeImageAttention, self).__init__()
        self.feature_size = image_feature_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.fc1_feature = nn.Conv2d(image_feature_size, hidden_size, kernel_size=1, bias=True)
        self.fc1_context = nn.Linear(context_size, hidden_size, bias=True)
        self.fc2 = nn.Conv2d(hidden_size, 1, kernel_size=1, bias=True)

    def forward(self, feature, context):
        batch_size = feature.shape[0]
        feature_hidden = self.fc1_feature(feature) # batch_size x hidden_size x width x height
        context_hidden = self.fc1_context(context) # batch_size x hidden_size
        context_hidden = context_hidden.unsqueeze(-2) # batch_size x 1 x hidden_size
        feature_hidden = feature_hidden.view(batch_size, self.hidden_size, -1) # batch_size x hidden_size x (width * height)
        x = torch.bmm(context_hidden, feature_hidden) # batch_size x 1 x (width x height)
        attention = F.softmax(x.view(batch_size, -1), 1).unsqueeze(-1) # batch_size x (width * height) x 1
        reshaped_features = feature.view(batch_size, self.feature_size, -1) # batch_size x feature_size x (width * height)
        x = torch.bmm(reshaped_features, attention) # batch_size x
        return x.squeeze(-1), attention.squeeze(-1)

class BottomUpImageAttention(nn.Module):
    def __init__(self, context_size, object_embedding_size, attribute_embedding_size, hidden_size, num_objects, num_attributes, image_feature_size=2048):
        super(BottomUpImageAttention, self).__init__()
        self.context_size = context_size
        self.object_embedding_size = object_embedding_size
        self.attribute_embedding_size = attribute_embedding_size
        self.hidden_size = hidden_size
        self.num_objects = num_objects
        self.num_attributes = num_attributes
        self.feature_size = image_feature_size + object_embedding_size + attribute_embedding_size + 1 + 5

        self.object_embedding = nn.Embedding(num_objects, object_embedding_size)
        self.attribute_embedding = nn.Embedding(num_attributes, attribute_embedding_size)

        self.fc1_context = nn.Linear(context_size, hidden_size)
        self.fc1_feature = nn.Linear(self.feature_size, hidden_size)
        #self.fc1 = nn.Linear(context_size + self.feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, bottom_up_features, context):
        # image_features: batch_size x max_num_detections x feature_size
        # object_ids: batch_size x max_num_detections
        # attribute_ids: batch_size x max_num_detections
        # no_object_mask: batch_size x max_num_detections
        # context: batch_size x context_size
        attribute_embedding = self.attribute_embedding(bottom_up_features.attribute_indices) # batch_size x max_num_detections x embedding_size
        object_embedding = self.object_embedding(bottom_up_features.object_indices) # batch_size x max_num_detections x embedding_size
        feats = torch.cat((bottom_up_features.cls_prob.unsqueeze(2), bottom_up_features.image_features, attribute_embedding, object_embedding, bottom_up_features.spatial_features), dim=2) # batch_size x max_num_detections x (feat size)

        # attended_feats = feats.mean(dim=1)
        # attention = None

        x_context = self.fc1_context(context).unsqueeze(1) # batch_size x 1 x hidden_size
        x_feature = self.fc1_feature(feats) # batch_size x max_num_detections x hidden_size
        x = x_context + x_feature # batch_size x max_num_detections x hidden_size
        x = F.relu(x)
        x = self.fc2(x).squeeze(-1) # batch_size x max_num_detections
        x.data.masked_fill_(bottom_up_features.no_object_mask, -float("-inf"))
        attention = F.softmax(x, 1).unsqueeze(1) # batch_size x 1 x max_num_detections
        attended_feats = torch.bmm(attention, feats).squeeze(1) # batch_size x feat_size
        return attended_feats, attention

class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, input_action_size, output_action_size, embedding_size, hidden_size,
                      dropout_ratio, feature_size=2048, image_attention_layers=None):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_action_size, embedding_size)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(hidden_size, output_action_size)
        self.image_attention_layers = image_attention_layers

    def forward(self, action, feature_list, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        action_embeds = self.embedding(action)   # (batch, 1, embedding_size)
        action_embeds = action_embeds.squeeze(1)

        inputs, attentions = [action_embeds], []
        for feature, attention_layer in zip(feature_list, self.image_attention_layers):
            if attention_layer:
                attended_feature, attention = attention_layer(feature, h_0)
            else:
                attended_feature = feature
                attention = None
            inputs.append(attended_feature)
            attentions.append(attention)

        concat_input = torch.cat(inputs , 1) # (batch, embedding_size+feature_size)
        drop = self.drop(concat_input)
        h_1,c_1 = self.lstm(drop, (h_0,c_0))
        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
        logit = self.decoder2action(h_tilde)
        return h_1,c_1,alpha,attentions,logit

## speaker models

class SpeakerEncoderLSTM(nn.Module):
    def __init__(self, num_actions, action_embedding_size, world_embedding_size, hidden_size, padding_idx,
                 dropout_ratio, bidirectional=False, num_layers=1):
        super(SpeakerEncoderLSTM, self).__init__()
        self.num_actions = num_actions
        self.action_embedding_size = action_embedding_size
        self.word_embedding_size = world_embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_actions, action_embedding_size, padding_idx)
        self.lstm = nn.LSTM(action_embedding_size + world_embedding_size, hidden_size, self.num_layers,
                            batch_first=True, dropout=dropout_ratio,
                            bidirectional=bidirectional)
        self.encoder2decoder = nn.Linear(hidden_size * self.num_directions,
                                         hidden_size * self.num_directions
                                         )

    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        c0 = Variable(torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ), requires_grad=False)
        return try_cuda(h0), try_cuda(c0)

    def forward(self, action_inputs, world_state_embeddings, lengths):
        ''' Expects action indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        batch_size = action_inputs.size(0)
        assert batch_size == world_state_embeddings.size(0)

        action_embeds = self.embedding(action_inputs)   # (batch, seq_len, embedding_size)
        embeds = torch.cat([action_embeds, world_state_embeddings], dim=2)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(batch_size)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

        if self.num_directions == 2:
            h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
            c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
        else:
            h_t = enc_h_t[-1]
            c_t = enc_c_t[-1] # (batch, hidden_size)

        decoder_init = nn.Tanh()(self.encoder2decoder(h_t))

        ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
        ctx = self.drop(ctx)
        return ctx,decoder_init,c_t  # (batch, seq_len, hidden_size*num_directions)
        # (batch, hidden_size)


class SpeakerDecoderLSTM(nn.Module):

    def __init__(self, vocab_size, vocab_embedding_size, hidden_size, dropout_ratio):
        super(SpeakerDecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.vocab_embedding_size = vocab_embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, vocab_embedding_size)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(vocab_embedding_size, hidden_size)
        self.attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(hidden_size, vocab_size)

    def forward(self, previous_word, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        word_embeds = self.embedding(previous_word)   # (batch, 1, embedding_size)
        word_embeds = word_embeds.squeeze()

        drop = self.drop(word_embeds)
        h_1,c_1 = self.lstm(drop, (h_0,c_0))
        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
        logit = self.decoder2action(h_tilde)
        return h_1,c_1,alpha,logit
