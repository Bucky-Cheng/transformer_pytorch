
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import math
import copy 
import time
import gc
import random
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


# MODEL 

# FeedForwardNetwork 

class FFN(nn.Module):
    def __init__(self, state_size = 200, forward_expansion = 1, bn_size=100, dropout=0.2):
        super(FFN, self).__init__()
        self.state_size = state_size
        
        self.lr1 = nn.Linear(state_size, forward_expansion * state_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(bn_size)
        self.lr2 = nn.Linear(forward_expansion * state_size, state_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.relu(self.lr1(x))
        x = self.bn(x)
        x = self.lr2(x)
        return self.dropout(x)

FFN()


# Mask 

def future_mask(seq_length):
    future_mask = (np.triu(np.ones([seq_length, seq_length]), k = 1)).astype('bool')
    return torch.from_numpy(future_mask)

future_mask(5)



class TransformerBlock_en(nn.Module):
    def __init__(self, embed_dim, heads = 4, MAX_SEQ = 100, dropout = 0.1, forward_expansion = 1):
        super(TransformerBlock_en, self).__init__()
        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_normal_q = nn.LayerNorm(embed_dim)
        self.layer_normal_k = nn.LayerNorm(embed_dim)
        self.layer_normal_v = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, forward_expansion = forward_expansion, bn_size=MAX_SEQ-1, dropout=dropout)
        self.layer_normal_2 = nn.LayerNorm(embed_dim)
        

    def forward(self, query, key, value, att_mask):


        query = self.layer_normal_q(query)
        key = self.layer_normal_k(key)
        value = self.layer_normal_v(value)

        att_output, att_weight = self.multi_att(query, key, value, attn_mask=att_mask)
        att_output = self.dropout(att_output + query)
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        att_output = self.layer_normal_2(att_output)
        x = self.ffn(att_output)
        x = self.dropout(x + att_output)
        return x.squeeze(-1), att_weight
    
class TransformerBlock_de(nn.Module):
    def __init__(self, embed_dim = 256, heads_de = 4, MAX_SEQ = 100, dropout = 0.1, forward_expansion = 1):
        super(TransformerBlock_de, self).__init__()
        
        self.multi_att_1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads_de, dropout=dropout)
        self.multi_att_2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads_de, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.layer_normal_de_in = nn.LayerNorm(embed_dim)
        self.layer_normal_en_out = nn.LayerNorm(embed_dim)
        self.layer_normal_de_out = nn.LayerNorm(embed_dim)
        
        self.layer_normal_1 = nn.LayerNorm(embed_dim)

        self.ffn = FFN(embed_dim, forward_expansion = forward_expansion, bn_size=MAX_SEQ-1, dropout=dropout)
        
        

    def forward(self, de_in, en_out, att_mask):

        de_in = self.layer_normal_de_in(de_in)
        att_output, att_weight = self.multi_att_1(de_in, de_in, de_in, attn_mask=att_mask)
        att_output = self.dropout_1(att_output + de_in)

        en_out = self.layer_normal_en_out(en_out)
        att_output = self.layer_normal_de_out(att_output)
        en_output, en_weight = self.multi_att_2(att_output, en_out, en_out, attn_mask=att_mask)
        en_output = self.dropout_2(en_output + att_output)

        en_output = en_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        en_output = self.layer_normal_1(en_output)
        x = self.ffn(en_output)
        x = self.dropout_3(x+en_output)
        return x.squeeze(-1), att_weight
    
class Encoder(nn.Module):
    def __init__(self, total_ex, total_cat, embed_dim, heads_en, max_seq,  dropout, forward_expansion, num_layers):
        super(Encoder, self).__init__()
        self.embed_dim = embed_dim
        
        self.embedding_id = nn.Embedding(total_ex , embed_dim)
        self.pos_embedding = nn.Embedding(max_seq - 1, embed_dim)
        self.embedding_part = nn.Embedding(total_cat + 1, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock_en(embed_dim, heads = heads_en, MAX_SEQ = max_seq, dropout = dropout, forward_expansion = forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, e_id, part_id):
        device = e_id.device
        e_id = self.embedding_id(e_id)
        pos_id = torch.arange(e_id.size(1)).unsqueeze(0).to(device)
        pos_x = self.pos_embedding(pos_id)
        part_id = self.embedding_part(part_id)
        x = self.dropout(e_id + part_id + pos_x)
        x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
       
        for layer in self.layers:
            att_mask = future_mask(x.size(0)).to(device)
            x, att_weight = layer(x, x, x, att_mask=att_mask)
            x = x.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        return x, att_weight

class Decoder(nn.Module):
    def __init__(self, total_in , total_task , total_lag , total_p , heads_de, max_seq, embed_dim, dropout, forward_expansion, num_layers):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.embedding_in = nn.Embedding(total_in, embed_dim)
        self.embedding_task = nn.Embedding(total_task, embed_dim)
        self.embedding_lag = nn.Embedding(total_lag, embed_dim)
        #self.embedding_p = nn.Embedding(total_p, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq - 1, embed_dim)
        self.p_time_con = nn.Linear(1, embed_dim, bias=False)
        #self.embedding_part = nn.Embedding(total_cat+1, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock_de(embed_dim, heads_de = heads_de, MAX_SEQ = max_seq, dropout = dropout, forward_expansion = forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, de_id, de_task, de_time, p_time, en_out):
        device = de_id.device
        de_in = self.embedding_in(de_id)
        #de_task = self.embedding_task(de_task)
        p_time = p_time.unsqueeze(0).permute(1, 2, 0).float()
        de_time = self.embedding_lag(de_time)
        p_time = self.p_time_con(p_time)
        pos_id = torch.arange(de_in.size(1)).unsqueeze(0).to(device)
        pos_x = self.pos_embedding(pos_id)
        x = self.dropout(de_in + de_time + p_time + pos_x)
        x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        en_out = en_out.permute(1, 0, 2)
       
        for layer in self.layers:
            att_mask = future_mask(x.size(0)).to(device)
            x, att_weight = layer(x, en_out, att_mask=att_mask)
            x = x.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        return x, att_weight


class TransformerModel(nn.Module):
    def __init__(self, total_ex, total_cat, total_in, total_task, total_lag, total_p, embed_dim, heads_en, heads_de, max_seq, dropout, forward_expansion = 1, enc_layers=3, dec_layers=3):
        super(TransformerModel, self).__init__()
        
        self.encoder = Encoder(total_ex, total_cat, embed_dim, heads_en, max_seq, dropout, forward_expansion, num_layers=enc_layers)
        self.decoder = Decoder(total_in, total_task, total_lag, total_p, heads_de, max_seq, embed_dim, dropout, forward_expansion, num_layers=dec_layers)
        self.pred = nn.Linear(embed_dim, 1)
        
    def forward(self, e_id, part_id,  de_task, de_time, p_time, de_in,):
        en_y, att_weight = self.encoder(e_id, part_id)
        x, att_weight_de = self.decoder(de_in, de_task, de_time, p_time, en_y)
        x = self.pred(x)
        return x.squeeze(-1), att_weight






