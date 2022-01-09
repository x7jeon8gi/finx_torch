from typing import ForwardRef
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from torch.autograd import Variable

def subsequent_mask(size):
    atten_shape = (1, size, size)
    mask = np.triu(np.ones(atten_shape), k=1).astype('uint8') # upper만 1이고 나머진 죄다 0
    return torch.from_numpy(mask) == 0 # 0인 부분만 True, 즉 (masking = False, non-masking = True)

def make_std_mask(tgt, pad):
    tgt_mask = (tgt !=pad) # pad masking
    tgt_mask = tgt_mask.unsqueeze(-2) # reshape (n_batch, seq_len) -> (n_batch, 1, seq_len)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)) 
    #beat-wise 연산자... 잘 모르겠다.
    return tgt_mask

class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, src_mask, trg_mask):
        encoder_output = self.encoder(src, src_mask)
        out = self.decoder(trg, trg_mask, encoder_output)

        return out

class Encoder(nn.Module):

    def __init__(self, encoder_layer, n_layer):
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_layer))
    
    def forward(self, x, mask):
        out = x
        for layer in self.layers:
            out = layer(out, mask)
        return out


class EncoderLayer(nn.Module):

    def __init__(self, multi_head_attention_layer, position_wise_feed_forward_layer, norm_layer):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention_layer = multi_head_attention_layer
        self.position_wise_feed_forward_layrer = position_wise_feed_forward_layer
        self.residual_connection_layers = [ResidualConnectionLayer(copy.deepcopy(norm_layer)) for i in range(2)]

    def forward(self, x, mask):
        out = self.residual_connection_layers[0](x, lambda x: self.multi_head_attention_layer(query=x, key=x, value=x, mask=mask))
        out = self.residual_connection_layers[1](x, lambda x: self.position_wise_feed_forward_layrer(out))
        return out


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, h , qkv_fc_layer, fc_layer):
        # qkv_fc_layer : (d_embed, d_model)
        # fc_layer : (d_model, d_embed)

        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.query_fc_layer = copy.deepcopy(qkv_fc_layer) # 서로 다른 weight로 학습되기 위해 deep copy 활용
        self.key_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.value_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.fc_layer = fc_layer

    def forward(self, query, key, value, mask=None):
        # q,k,v : (n_batch, seq_len, d_embed), mask: (n_batch, seq_len, seq_len)

        n_batch = query.shape[0]

        def transform(x, fc_layer): # 받은 문장의 d_embed를 d_k로 변환
            out = fc_layer(x) #(n_batch, seq_len, d_model) ## 근데 d_embed == d_model 이면 크게 문제 없지 않나? 원 논문도 $$_fc_layer 거치나?
            out = out.view(n_batch, -1, self.h, self.d_model//self.h) # (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)
            return out

        query = transform(query, self.query_fc_layer) # (n_batch, h, seq_len, d_k)
        key = transform(key, self.key_fc_layer)
        value = transform(value, self.value_fc_layer)

        if mask is not None:
            mask = mask.unsqueeze(1) # (n_batch, 1, seq_len, seq_len) head 부분에 차원을 추가하여 마스킹할 수 있도록 조절(broadcasting)

        out = self.calculate_attention(query, key, value, mask) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1,2) # (n_batch, seq_len, h, d_k)
        out = out.view(n_batch, -1, self.d_model)
        out = self.fc_layer(out) # (n_batch, seq_len, d_embed)
        return out

    def calculate_attention(self, query, key, value, mask):
        # (n_batch, seq_len, d_k)
        d_k = key.size(-1)
        attention_score = torch.matmul(query, key.transpose(-2,-1)) # (n_batch, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k) # scailing
        if mask is not None:
            attention_score = attention_score.masked_fill(attention_score==0, -1e9) # attention_score==0 인 부분을 모두 마스킹처리 (mask pad로 설정될 것)
        attention_prob = F.softmax(attention_score, dim=-1) #(n_batch, seq_len, seq_len)
        out = torch.matmul(attention_prob, value) # (n_batch, seq_len, d_k)
        return out

class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, first_fc_layer, second_fc_layer):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.first_fc_layer = first_fc_layer
        self.second_fc_layer = second_fc_layer
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.first_fc_layer(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.second_fc_layer(out)
        
        return out

class ResidualConnectionLayer(nn.Module):

    def __init__(self, multi_head_attention_layer, position_wise_feed_forward_layer, norm_layer):
        super(ResidualConnectionLayer, self).__init__()
        self.norm_layer = norm_layer

    def forward(self, x ,sub_layer):
        out = sub_layer(x) + x
        out = self.norm_layer(out)
        return out


class Decoder(nn.Module):
    def __init__(self, sub_layer, n_layer):
        super(Decoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(sub_layer))

    def forward(self, x, mask, encoder_output, encoder_mask):
        out = x
        for layer in self.layers:
            out = layer(out, mask, encoder_output, encoder_mask)

        return out

class DecoderLayer(nn.Module):
    