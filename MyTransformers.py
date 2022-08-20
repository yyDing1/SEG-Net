import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.nn import LayerNorm

import copy
import math
import random
import numpy as np



class Selector(nn.Module):
    def __init__(self, d_model, num_head, d_ff, dropout, layer_num, vocab_size):
        super(Selector, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadedAttention(num_head, d_model, normal_attention, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.embedding = SEGEmbedding(vocab_size, d_model, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), layer_num)
        self.maxout = Maxout(d_model * 2, 1, 3)
        
    def forward(self, x, seg):
        src_mask = (x != 0).unsqueeze(-2).long()
        mask = (x != 0).long()
        pos = torch.arange(x.size(-1)).unsqueeze(0).to(x.device)
        x = self.embedding(x, pos, seg.unsqueeze(-1))
        x = self.encoder(x, src_mask)
        mean_pool = (x * mask.unsqueeze(-1)).sum(dim=-2) / mask.sum(dim=-1).unsqueeze(-1)
        max_pool = (x + (1 - mask.unsqueeze(-1)) * -1e9).max(dim=-2).values
        pooling = torch.cat((mean_pool, max_pool), dim=-1)
        output = self.maxout(pooling)
        return output

    def init_params(self):
        for name, param in self.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass


class Maxout(nn.Module):
    def __init__(self, d_in, d_out, layers=3):
        super(Maxout, self).__init__()
        self.layers = clones(nn.Linear(d_in, d_out), layers)
        self.batch_norm = torch.nn.BatchNorm1d(d_out)
        
    def forward(self, x):
        max_output = self.batch_norm(self.layers[0](x))
        for _, layer in enumerate(self.layers, start=1):
            max_output = self.batch_norm(torch.max(max_output, layer(x)))
        return max_output


class ExtractorGenerator(nn.Module):
    def __init__(self, d_model, num_head, d_ff, dropout, layer_num, vocab_size):
        super(ExtractorGenerator, self).__init__()
        c = copy.deepcopy
        self_attn = MultiHeadedAttention(num_head, d_model, normal_attention, dropout)
        cov_attn = MultiHeadedAttention(num_head, d_model, coverage_attention, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.embedding = SEGEmbedding(vocab_size, d_model, dropout)
        self.layer_wise_encoder_decoder = LayerWiseEncoderDecoder(
            EncoderLayer(d_model, c(self_attn), c(ff), dropout),
            DecoderLayer(d_model, c(self_attn), c(cov_attn), c(ff), dropout),
            layer_num
        )
        self.extractor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.copy_attn = CopyAttention(d_model, vocab_size)
    
    def forward(self, src, tgt, copy_mask, segment_id):
        src_mask = (src != 0).unsqueeze(-2).long()
        tgt_mask = (tgt != 0).unsqueeze(-2).long()
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data).to(src_mask.device)
        src_pos = torch.arange(src.size(-1)).unsqueeze(0).to(src.device)
        src_emb, tgt_emb = self.embedding(src, src_pos, segment_id), self.embedding(tgt)
        src_output, tgt_output = self.layer_wise_encoder_decoder(src_emb, src_mask, tgt_emb, tgt_mask)
        extractor_output = self.extractor(src_output)
        generator_output, copy_output = self.copy_attn(src_output, tgt_output, copy_mask)
        return extractor_output, generator_output, copy_output
    
    def init_params(self):
        for name, param in self.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass


class CopyAttention(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(CopyAttention, self).__init__()
        self.W_att = nn.Linear(d_model, d_model, bias=False)
        self.W_u = nn.Linear(d_model * 2, 1)
        self.proj = nn.Linear(d_model, vocab_size)
        self.nllloss = nn.NLLLoss()

    def forward(self, src, tgt, copy_mask):
        a = torch.bmm(self.W_att(tgt), src.transpose(-2, -1)).softmax(dim=-1)
        c = torch.bmm(a, src)
        prob_copy = torch.sigmoid(self.W_u(torch.cat((tgt, c), dim=-1))).squeeze(-1)  # (batch_size, tgt_len)
        copy_prob = a  # (batch_size, tgt_len, src_len)
        generator_output = self.proj(tgt).softmax(dim=-1)  # (batch_size, tgt_len, vocab_size)
        # copy_mask: (batch_size, tgt_len, src_len)
        copy_output = (copy_mask * copy_prob).sum(dim=-1)  # (batch_size, tgt_len), max or sum
        return (1 - prob_copy).unsqueeze(-1) * generator_output, prob_copy * copy_output


class ExtractorGeneratorLoss(nn.Module):
    def __init__(self, w, beta, vocab_size):
        super(ExtractorGeneratorLoss, self).__init__()
        self.w = w
        self.beta = beta
        self.vocab_size = vocab_size
        self.loss_func_e = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w), reduction='none')
        self.nllloss = nn.NLLLoss(reduction='none')
    
    def forward(self, extractor_output, generator_output, copy_output, extractor_target, generator_target, src_mask, tgt_mask):
        # loss_extractor
        loss_e = self.loss_func_e(extractor_output, extractor_target.float().unsqueeze(-1)).squeeze(-1)  # (batch_size, src_len)
        loss_e = loss_e.sum(dim=-1) / src_mask.sum(dim=-1)  # (batch_size, )

        # loss_generator
        # generator_output: (batch_size, tgt_len, vocab)
        # copy_output: (batch_size, tgt_len)
        # generator_target: (batch_size, tgt_len)
        generator_output[:, :, 1] = 1
        no_copy_ouput = (generator_output * F.one_hot(generator_target, num_classes=self.vocab_size)).sum(dim=-1)
        # print(no_copy_ouput)
        loss_g = -(no_copy_ouput + copy_output).log()  # (batch_size, tgt_len)
        loss_g = loss_g.sum(dim=-1) / tgt_mask.sum(dim=-1)  # (batch_size, )
        loss = loss_e * self.beta + loss_g * (1 - self.beta)
        return loss_g.mean()


class SEGEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, max_len=5000, max_seg=1000):
        super(SEGEmbedding, self).__init__()
        self.d_model = d_model
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(max_seg, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, pos=None, seg=None):
        output = self.word_embedding(x) * math.sqrt(self.d_model)
        if pos is not None:
            output += self.position_embedding(pos)
        if seg is not None:
            output += self.segment_embedding(seg)
        return self.dropout(output)


class LayerWiseEncoderDecoder(nn.Module):
    def __init__(self, encoder_layer, decoder_layer, N):
        super(LayerWiseEncoderDecoder, self).__init__()
        self.layer_num = N
        self.encoder_layers = clones(encoder_layer, N)
        self.decoder_layers = clones(decoder_layer, N)
        self.norm = clones(LayerNorm(encoder_layer.size), N + 1)
        
    def forward(self, src_input, src_mask, tgt_input, tgt_mask):
        src, tgt = src_input, tgt_input
        for i in range(self.layer_num):
            encoder_layer, decoder_layer, norm = self.encoder_layers[i], self.decoder_layers[i], self.norm[i]
            src = encoder_layer(src, src_mask)
            tgt = decoder_layer(tgt, norm(src), src_mask, tgt_mask)
        return self.norm[-2](src), self.norm[-1](tgt)
        

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0
    
def normal_attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def coverage_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    scores = torch.exp(scores)
    # mask: (batch_size, head, seq_len, seq_len)
    tgt_seq_len = scores.size()[-2]
    prefix_sum = 1 - torch.triu(torch.ones((tgt_seq_len - 1, tgt_seq_len - 1)), diagonal=1).unsqueeze(0).unsqueeze(0)
    prefix_sum = prefix_sum.to(scores.device)
    cov_scores = torch.zeros_like(scores)
    cov_scores[:, :, 0] = scores[:, :, 0]
    cov_scores[:, :, 1:] = scores[:, :, 1:] / torch.matmul(prefix_sum, scores[:, :, :-1])
    if mask is not None:
        cov_scores = cov_scores.masked_fill(mask == 0, 0)
    p_attn = cov_scores / cov_scores.sum(dim=-1, keepdim=True)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, attention_func, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.attention_func = attention_func
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention_func(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
