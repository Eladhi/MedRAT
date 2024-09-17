from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .att_model import pack_wrapper, AttModel


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32, temperature=1.0):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    selected_scores, idx = scores.topk(topk)
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1))
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1))
    selected_value = torch.gather(dummy_value, 3, dummy_idx)
    p_attn = F.softmax(selected_scores / temperature, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn, idx


class Transformer(nn.Module):
    def __init__(self, encoder_src, encoder_tgt, joint_encoder, decoder, src_embed, tgt_embed, cmn, self_attn, noise, unsupervised, use_glob_feat, lg_embedding, decoder_dropout):
        super(Transformer, self).__init__()
        self.encoder_src = encoder_src
        self.encoder_tgt = encoder_tgt
        self.joint_encoder = joint_encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.cmn = cmn
        self.self_attn = self_attn
        self.noise_std = noise
        self.unsupervised = unsupervised
        self.use_glob_feat = use_glob_feat
        self.lg_embedding = lg_embedding
        self.decoder_dropout = decoder_dropout

    def forward(self, src, tgt, src_mask, tgt_mask, memory_matrix, mode='forward'):  # only for train; at infer it's prepare features & core
        src_encoded, _ = self.get_src_encoding(src, src_mask)
        src_glob = self.get_glob_feat(src_encoded, src_mask)
        tgt_emb = self.get_seq_embed(tgt, memory_matrix)
        tgt_encoded, _ = self.get_tgt_encoding(tgt_emb, tgt_mask)
        tgt_glob = self.get_glob_feat(tgt_encoded, tgt_mask)
        if self.use_glob_feat:
            src_encoded = torch.concat((src_glob.clone().unsqueeze(1), src_encoded), dim=1)
            src_type = torch.ones((src_encoded.shape[0], src_encoded.shape[1])).int().to(src_encoded.device)
            src_type[:, 0] *= 0
            src_encoded = src_encoded + self.lg_embedding(src_type)
            tgt_encoded = torch.concat((tgt_glob.clone().unsqueeze(1), tgt_encoded), dim=1)  # addition of global features
            tgt_type = torch.ones((tgt_encoded.shape[0], tgt_encoded.shape[1])).int().to(tgt_encoded.device)
            tgt_type[:, 0] *= 0
            tgt_encoded = tgt_encoded + self.lg_embedding(tgt_type)

        if mode == 'forward' and self.unsupervised:
            _src_mask = tgt_mask[:, -1].unsqueeze(1)
            if self.use_glob_feat:
                _src_mask = torch.concat((torch.ones((_src_mask.shape[0], 1, 1)).byte().to(_src_mask.device), _src_mask), dim=2)

            noise = torch.normal(torch.zeros_like(tgt_encoded), self.noise_std * torch.ones_like(tgt_encoded))
            if self.use_glob_feat:
                noise[:, 0] *= 0
            _encoded_s = tgt_encoded.clone() + noise

            # dropout
            if self.decoder_dropout > 0:
                keep_idx = torch.rand((_src_mask.shape[0], _src_mask.shape[-1])) < (1 - self.decoder_dropout)
                keep_idx = keep_idx.to(_src_mask.device)
                if self.use_glob_feat:
                    keep_idx[:, 0] = True
                _src_mask = _src_mask * keep_idx.unsqueeze(1)
                _encoded_s = _encoded_s * keep_idx.unsqueeze(2).repeat([1, 1, _encoded_s.shape[-1]])

        else:
            _src_mask = src_mask
            if self.use_glob_feat:
                _src_mask = torch.concat((torch.ones((_src_mask.shape[0], 1, 1)).byte().to(_src_mask.device), _src_mask), dim=2)
            _encoded_s = src_encoded.clone()
        return self.decode(_encoded_s, _src_mask, tgt_emb, tgt_mask), src_glob, tgt_glob

    def get_src_encoding(self, x, mask):
        src_encoding, p_attn = self.encoder_src(self.src_embed(x), mask)
        joint_encoding, p_attn_joint = self.joint_encoder(src_encoding, mask)
        return joint_encoding, torch.matmul(p_attn_joint, p_attn)

    def get_tgt_encoding(self, x, mask):
        tgt_encoding, p_attn = self.encoder_tgt(x, mask[:, -1].unsqueeze(1))
        tgt_encoded, p_attn_joint = self.joint_encoder(tgt_encoding, mask[:, -1].unsqueeze(1))
        return tgt_encoded, torch.matmul(p_attn_joint, p_attn)

    def get_seq_embed(self, x, memory_matrix):
        tgt_emb = self.tgt_embed[:-1](x)
        tgt_emb = self.tgt_memory(tgt_emb, memory_matrix)
        tgt_emb = self.tgt_embed[-1](tgt_emb)
        return tgt_emb

    def decode_infer(self, memory, mask, ys, past, memory_matrix):
        if self.use_glob_feat:
            memory, mask, glob_weights = self.add_glob_feat(memory, mask, return_weights=True)
            enc_g = memory[:, 0].detach().clone()
        else:
            enc_g = self.get_glob_feat(memory, mask)
        out, past, p_attn = self.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device),
                                past=past,
                                memory_matrix=memory_matrix)
        if self.use_glob_feat:
            glob_portion = p_attn[:, :, 0].repeat((1, glob_weights.shape[1])) * glob_weights
            p_attn = p_attn[:, :, 1:].squeeze(1) + glob_portion
            p_attn = p_attn.unsqueeze(1)
            p_attn = torch.concat((p_attn, glob_weights.unsqueeze(1)), dim=1)
        return out, past, enc_g, p_attn

    def get_glob_feat(self, x, mask, return_weights=False):
        return self.self_attn(x, mask, return_weights)

    def add_glob_feat(self, x, mask, return_weights=False):
        g = self.get_glob_feat(x, mask, return_weights)
        if return_weights:
            g, weights = g[0], g[1]
        x = torch.concat((g.unsqueeze(1), x), dim=1)
        x_type = torch.ones((x.shape[0], x.shape[1])).int().to(x.device)
        x_type[:, 0] *= 0
        x = x + self.lg_embedding(x_type)
        mask = torch.concat((torch.ones((mask.shape[0], 1, 1)).byte().to(x.device), mask), dim=2)
        if return_weights:
            return x, mask, weights
        return x, mask

    def decode(self, memory, src_mask, tgt_emb, tgt_mask, past=None, memory_matrix=None):
        if past is not None:
            tgt_emb = self.get_seq_embed(tgt_emb, memory_matrix)
        return self.decoder(tgt_emb, memory, src_mask, tgt_mask, past=past)

    def tgt_memory(self, embeddings, memory_matrix=None):
        # Memory querying and responding for textual features
        dummy_memory_matrix = memory_matrix.unsqueeze(0).expand(embeddings.size(0), memory_matrix.size(0),
                                                                memory_matrix.size(1))
        responses, mem_idx = self.cmn(embeddings, dummy_memory_matrix, dummy_memory_matrix)
        embeddings = embeddings + responses
        self.mem_idx = mem_idx
        return embeddings


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        p_attn = None
        for li, layer in enumerate(self.layers):
            x = layer(x, mask)
            if type(x) is tuple:
                x, attn_l = x[0], x[1]
                if li == 0:
                    p_attn = attn_l
                else:
                    p_attn = torch.matmul(attn_l, p_attn)
        return self.norm(x), p_attn


class LayerNorm(nn.Module):
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
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        _x = sublayer(self.norm(x))
        # if type(_x) is tuple:
        #     return x + self.dropout(_x[0]), _x[1]
        if type(_x) is tuple and len(_x) == 2:
            return x + self.dropout(_x[0]), _x[1]
        elif type(_x) is tuple and len(_x) == 3:
            return x + self.dropout(_x[0]), _x[1], _x[2]
        return x + self.dropout(_x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        if type(x) is tuple and len(x) == 2:
            return self.sublayer[1](x[0], self.feed_forward), x[1]
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, past=None):
        if past is not None:
            present = [[], []]
            x = x[:, -1:]
            tgt_mask = tgt_mask[:, -1:] if tgt_mask is not None else None
            past = list(zip(past[0].split(2, dim=0), past[1].split(2, dim=0)))
        else:
            past = [None] * len(self.layers)
        p_attn = None
        for i, (layer, layer_past) in enumerate(zip(self.layers, past)):
            x = layer(x, memory, src_mask, tgt_mask,
                      layer_past)
            if type(x) is tuple and len(x) == 3:
                attn_l = x[2]
                if i == 0:
                    p_attn = attn_l / len(self.layers)
                else:
                    p_attn += (attn_l / len(self.layers))
            if layer_past is not None:
                present[0].append(x[1][0])
                present[1].append(x[1][1])
                x = x[0]
        if past[0] is None:
            return self.norm(x)
        else:
            return self.norm(x), [torch.cat(present[0], 0), torch.cat(present[1], 0)], p_attn


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, layer_past=None):
        m = memory
        if layer_past is None:
            x, _ = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x, _ = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward)
        else:
            present = [None, None]
            x, present[0], _ = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0]))
            x, present[1], p_attn = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1]))
            return self.sublayer[2](x, self.feed_forward), present, p_attn


class MultiThreadMemory(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, topk=32, temperature=1.0, kq_normalize=True):
        super(MultiThreadMemory, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.topk = topk
        self.temperature = temperature
        self.kq_normalize = kq_normalize

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]
        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        # normalize kq
        if self.kq_normalize:
            query = F.normalize(query, p=2, dim=-1)
            key = F.normalize(key, p=2, dim=-1)

        x, self.attn, mem_idx = memory_querying_responding(query, key, value, mask=mask,
                                                           dropout=self.dropout, topk=self.topk,
                                                           temperature=self.temperature)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present, mem_idx
        else:
            return self.linears[-1](x), mem_idx


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        x_attn = self.attn.mean(1)
        if layer_past is not None:
            # return self.linears[-1](x), present
            return self.linears[-1](x), present, x_attn
        else:
            # return self.linears[-1](x)
            return self.linears[-1](x), x_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SelfAttn(nn.Module):
    def __init__(self, d_model, dropout=0.1, normalize_output=False):
        super(SelfAttn, self).__init__()
        self.s1 = nn.Sequential(nn.Linear(d_model, d_model),
                                              nn.Tanh(), nn.Dropout(dropout))
        self.s2 = nn.Sequential(nn.Linear(d_model, 1))
        self.softmax = nn.Softmax(dim=1)
        self.normalize_output = normalize_output

    def forward(self, x, mask, return_weights=False):
        mask = torch.stack(x.shape[-1] * [mask[:, -1]], dim=2)
        x *= mask

        s1_out = self.s1(x)
        s1_out = x.mul(s1_out)
        s2_out = self.s2(s1_out).squeeze(2)
        weights = self.softmax(s2_out)

        # compute final image
        x = (weights.unsqueeze(2) * x).sum(dim=1)
        # normalize
        if self.normalize_output:
            x = F.normalize(x)

        if return_weights:
            return x, weights
        return x

class Classifier(nn.Module):
    def __init__(self, d_model, d_ff, n_classes, dropout=0.1):
        super(Classifier, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return torch.sigmoid(self.w_2(self.dropout(F.relu(self.w_1(x)))))


class BaseCMN(AttModel):

    def make_model(self, tgt_vocab, cmn, args):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers_shared),
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.num_layers),
            nn.Sequential(c(position)),
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), nn.Sequential(c(position))),
            cmn, SelfAttn(self.d_model, normalize_output=True), args.noise_std, args.unsupervised,
            args.use_glob_feat, nn.Embedding(2, self.d_model), args.decoder_dropout)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(BaseCMN, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers
        self.num_layers_shared = args.num_layers_shared
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.topk = args.topk
        self.n_classes = args.n_pathologies
        self.unsupervised = args.unsupervised

        tgt_vocab = self.vocab_size + 1

        self.cmn = MultiThreadMemory(args.num_heads, args.d_model, topk=args.topk, temperature=args.smax_temperature,
                                     kq_normalize=args.kq_normalize)

        self.model = self.make_model(tgt_vocab, self.cmn, args)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

        self.memory_matrix = nn.Parameter(torch.FloatTensor(args.cmm_size, args.cmm_dim))
        nn.init.normal_(self.memory_matrix, 0, 1 / args.cmm_dim)
        self.log_memory_histograms = False
        self.memory_histogram_src = torch.zeros(args.cmm_size, dtype=torch.int64)
        self.memory_histogram_tgt = torch.zeros(args.cmm_size, dtype=torch.int64)

        self.classifier = Classifier(self.d_model, self.d_ff, self.n_classes, dropout=self.dropout)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):  # called only on inference
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory, p_attn = self.model.get_src_encoding(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks, p_attn

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)

        # Memory querying and responding for visual features
        dummy_memory_matrix = self.memory_matrix.unsqueeze(0).expand(att_feats.size(0), self.memory_matrix.size(0),
                                                                     self.memory_matrix.size(1))
        responses, mem_idx = self.cmn(att_feats, dummy_memory_matrix, dummy_memory_matrix)
        att_feats = att_feats + responses

        if self.log_memory_histograms:
            _mem_idx = mem_idx
            mem_idx, mem_cnt = _mem_idx.unique(return_counts=True)[0], _mem_idx.unique(return_counts=True)[1]
            self.memory_histogram_src.index_add_(0, mem_idx, mem_cnt)
        # Memory querying and responding for visual features

        att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, mode='train'):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        out, enc_s, enc_t = self.model(att_feats, seq, att_masks, seq_mask, memory_matrix=self.memory_matrix, mode=mode)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        src_class = self.classifier(enc_s.clone())
        tgt_class = self.classifier(enc_t.clone())

        if self.log_memory_histograms:
            _mem_idx = self.model.mem_idx
            mem_idx, mem_cnt = _mem_idx.unique(return_counts=True)[0], _mem_idx.unique(return_counts=True)[1]
            self.memory_histogram_tgt.index_add_(0, mem_idx, mem_cnt)

        return outputs, src_class, tgt_class, enc_s, enc_t

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask, p_attn_feat):
        if len(state) == 0:
            ys = it.unsqueeze(1)
            past = [fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model),
                    fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model)]
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
            past = state[1:]
        out, past, enc_s, p_attn_dec = self.model.decode_infer(memory, mask, ys, past=past,
                                             memory_matrix=self.memory_matrix)

        if p_attn_dec.shape[2] > p_attn_feat.shape[1]:
            p_attn_dec = p_attn_dec[:, :, 1:]
        attn_weights = p_attn_dec.transpose(1, 2)

        if self.log_memory_histograms:
            _mem_idx = self.model.mem_idx
            mem_idx, mem_cnt = _mem_idx.unique(return_counts=True)[0], _mem_idx.unique(return_counts=True)[1]
            self.memory_histogram_tgt.index_add_(0, mem_idx, mem_cnt)

        src_class = self.classifier(enc_s)
        return out[:, -1], [ys.unsqueeze(0)] + past, [src_class, attn_weights]
