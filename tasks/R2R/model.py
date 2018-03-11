
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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

    def init_state(self, inputs):
        ''' Initialize to zero cell states and hidden states.'''
        batch_size = inputs.size(0)
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
        return h0.cuda(), c0.cuda()

    def forward(self, inputs, lengths):
        ''' Expects input vocab indices as (batch, seq_len). Also requires a 
            list of lengths for dynamic batching. '''
        embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)
        h0, c0 = self.init_state(inputs)
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
        self.sm = nn.Softmax()
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
    def __init__(self, feature_size, context_size, attention_size):
        super(FeedforwardImageAttention, self).__init__()
        self.feature_size = feature_size
        self.context_size = context_size
        self.attention_size = attention_size
        self.fc1_feature = nn.Conv2d(feature_size, attention_size, kernel_size=1, bias=False)
        self.fc1_context = nn.Linear(context_size, attention_size, bias=True)
        self.fc2 = nn.Conv2d(attention_size, 1, kernel_size=1, bias=True)

    def forward(self, feature, context):
        batch_size = feature.shape[0]
        feature_hidden = self.fc1_feature(feature)
        context_hidden = self.fc1_context(context)
        context_hidden = context_hidden.unsqueeze(-1).unsqueeze(-1)
        x = feature_hidden + context_hidden
        x = self.fc2(F.relu(x))
        x = F.softmax(x.view(batch_size, -1), 1).unsqueeze(-1) # batch_size x (width * height) x 1
        reshaped_features = feature.view(batch_size, self.feature_size, -1) # batch_size x feature_size x (width * height)
        x = torch.bmm(reshaped_features, x) # batch_size x
        return x.squeeze(-1)

class MultiplicativeImageAttention(nn.Module):
    def __init__(self, feature_size, context_size, attention_size):
        super(MultiplicativeImageAttention, self).__init__()
        self.feature_size = feature_size
        self.context_size = context_size
        self.attention_size = attention_size
        self.fc1_feature = nn.Conv2d(feature_size, attention_size, kernel_size=1, bias=True)
        self.fc1_context = nn.Linear(context_size, attention_size, bias=True)
        self.fc2 = nn.Conv2d(attention_size, 1, kernel_size=1, bias=True)

    def forward(self, feature, context):
        batch_size = feature.shape[0]
        feature_hidden = self.fc1_feature(feature) # batch_size x attention_size x width x height
        context_hidden = self.fc1_context(context) # batch_size x attention_size
        context_hidden = context_hidden.unsqueeze(-2) # batch_size x 1 x attention_size
        feature_hidden = feature_hidden.view(batch_size, self.attention_size, -1) # batch_size x attention_size x (width * height)
        x = torch.bmm(context_hidden, feature_hidden) # batch_size x 1 x (width x height)
        x = F.softmax(x.view(batch_size, -1), 1).unsqueeze(-1) # batch_size x (width * height) x 1
        reshaped_features = feature.view(batch_size, self.feature_size, -1) # batch_size x feature_size x (width * height)
        x = torch.bmm(reshaped_features, x) # batch_size x
        return x.squeeze(-1)

class AttnDecoderLSTM(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, input_action_size, output_action_size, embedding_size, hidden_size,
                      dropout_ratio, feature_size=2048, ablate_image_features=False, image_attention_layer=None):
        super(AttnDecoderLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_action_size, embedding_size)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(hidden_size, output_action_size)
        self.ablate_image_features = ablate_image_features
        if image_attention_layer:
            self.image_attention_layer = image_attention_layer

    def forward(self, action, feature, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        action_embeds = self.embedding(action)   # (batch, 1, embedding_size)
        action_embeds = action_embeds.squeeze()
        if self.ablate_image_features:
            feature = torch.zeros_like(feature)
        if self.image_attention_layer:
            feature = self.image_attention_layer(feature, h_0)

        concat_input = torch.cat((action_embeds, feature), 1) # (batch, embedding_size+feature_size)
        drop = self.drop(concat_input)
        h_1,c_1 = self.lstm(drop, (h_0,c_0))
        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
        logit = self.decoder2action(h_tilde)
        return h_1,c_1,alpha,logit
