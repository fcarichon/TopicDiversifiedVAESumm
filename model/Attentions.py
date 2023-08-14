#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")

# Seed pour reproduire les résultats.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class Group_Attention(nn.Module):
    
    def __init__(self, group_dim, attn_dim, output_dim=1):
        """Attention for the group representation | Latent variable c"""
        super().__init__()
        self.attn = nn.Linear(group_dim, attn_dim)
        self.v = nn.Linear(attn_dim, output_dim, bias = False)
    
    def forward(self, group_encodings, group_mask):
        """
        Args:
            group_encodings = [batch_size, total_elements=gpe_size * seq_len, group_dim=enc_dim + embedd_dim]
            group_mask = [batch_size, total_elements]
        Output:
            F.softmax(attention, dim=1) : Attention distribution tensor for whole group 
        """
        batch_size = group_encodings.shape[0]
        tot_els = group_encodings.shape[1]
        
        #Calculation of attention vector
        group_energy = torch.tanh(self.attn(group_encodings))                             #[batch size, gpe_size * seq_len, attn_dim]
        attention = self.v(group_energy).squeeze(2)                                       #[batch size, gpe_size * seq_len]
        
        #Masking padded tokens when calculation of attention
        attention = attention.masked_fill(group_mask == 0, -1e10)                         #[batch size, gpe_size * seq_len]
        
        return F.softmax(attention, dim=1)
    
    
class Attention(nn.Module):
    """Attention for the individual representation | decoding"""
    def __init__(self, enc_dim, dec_dim, attn_dim, output_dim=1):
        super().__init__()
        self.enc_dim = enc_dim
        self.enc_features = nn.Linear(enc_dim, attn_dim)
        self.hid_features = nn.Linear(dec_dim, attn_dim)
        self.attn = nn.Linear(attn_dim, attn_dim)
        self.v = nn.Linear(attn_dim, output_dim, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask):
        """
        hidden = [1, batch_size, dec_dim]
        encoder_outputs = [batch_size, seq_len, enc_dim]
        mask = [batch_size, seq_len]
        """
        src_len = encoder_outputs.shape[1]
        hidden = hidden.permute((1, 0, 2))                                                 #hidden = [batch_size, 1, dec_dim]
        
        #repeat decoder hidden state seq_len times
        hidden = hidden[:, -1, :].unsqueeze(1).repeat(1, src_len, 1)                       #[batch_size, seq_len, dec_dim]
        decoding_features = self.hid_features(hidden)                                      #[batch_size, seq_len, dec_dim]
        encoding_features = self.enc_features(encoder_outputs)                             #[batch_size, seq_len, dec_dim]
        att_features = decoding_features + encoding_features
        energy = torch.tanh(self.attn(att_features))                                       #[batch size, src len, dec_dim]
        
        #Calculation of attention vector
        attention = self.v(energy).squeeze(2)                                              #[batch size, src len]
        attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim=1)