#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F

# Seed pour reproduire les résultats.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

from model.Attentions import Attention


class GRU_Decoder(nn.Module):
    """Decoding hidden representations into token sequence"""
    def __init__(self, output_dim, emb_vecs, emb_dim, enc_dim, dec_dim, attn_dim, z_dim, dropout, context_vector_input=False, add_extrafeat=False, use_topic_attention=False):
        super().__init__()
        """
        Args:
            Dimension args
            context_vector_input: To account or not for contextual vector in decoding
            add_extrafeat: To account or not for additional extra_feature in decoding
        """
        #### Model parameters
        self.emb_vecs = emb_vecs
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.attn_dim = attn_dim
        self.z_dim = z_dim
        self.dropout = nn.Dropout(dropout)
        self.context_vector_input = context_vector_input
        self.add_extrafeat = add_extrafeat
        self.use_topic_attention = use_topic_attention
        
        #### Model layers
        self.attention_layer = Attention(enc_dim, dec_dim, attn_dim)
        #With adding context vector as input - put option
        if self.context_vector_input:
            if self.add_extrafeat:
                self.rnn = nn.GRU(emb_dim + z_dim + enc_dim, dec_dim, dropout = dropout, bidirectional=False)
            else:
                self.rnn = nn.GRU(emb_dim + enc_dim, dec_dim, dropout = dropout, bidirectional=False)
        else:
            if self.add_extrafeat:
                self.rnn = nn.GRU(emb_dim + z_dim, dec_dim, dropout = dropout, bidirectional=False)
            else:
                self.rnn = nn.GRU(emb_dim, dec_dim, dropout = dropout, bidirectional=False)
                
        self.fc_out = nn.Linear(dec_dim + enc_dim, output_dim)
        self.fc_out_inter = nn.Linear(dec_dim + enc_dim, emb_dim)
        
        #Adaptative Softmax with 4 clusters
        self.out_ada=nn.AdaptiveLogSoftmaxWithLoss(dec_dim + enc_dim + emb_dim, output_dim, cutoffs=[round(output_dim/12), 3*round(output_dim/12)], div_value=4)
        
    def forward(self, input_embs, enc_outputs, hidden, mask_others, src_mask, context_weight, extra_feat, topic_elem):
        """
        Args:
            input_embs = [batch_size, emb_dim] embeddings of current words to decode
            encoder_outputs = [batch_size, tot_elem, enc_dim]
            src_mask = [batch_size, 1]
            mask_others = [batch_size, total_elem]
            hidden = [batch_size, 1, dec_dim]
            context_vector = [batch_size, enc_dim*2] | [batch_size, enc_dim]
            extra_feat = [batch_size, dec_dim = z_dim + num_topics]
            topic_elem  = [batch_size, total_elem]
            use_topic_attention: If you want to ponder attention in decoding step with the beta distribution -- topic_elem
        Output:
            vocab_dist : Distribution over the vocabulary for next token prediction
            attn_dist : Attention distribution for PGN weighting
            context_weight_new : context vector updated
            hidden_new : new hidden layer for the current sentence accounting for the new token
        """
        rnn_input = input_embs
        hidden = hidden.unsqueeze(0)                                                               # [1, batch_size, dec_dim = z_dim + num_topics]
        hidden = hidden.contiguous()
        
        #Apply attention layer to output attn_distribution -- we attend to all other word from group when decoding here
        attn_dist = self.attention_layer(hidden, enc_outputs, mask_others)                         # [batch size, tot_elem]
        if self.use_topic_attention:
            attn_dist_ = topic_elem * attn_dist
            a = attn_dist_.unsqueeze(1)
        else:
            a = attn_dist.unsqueeze(1)                                                             # [batch size, 1, tot_elem]
        
        
        context_weight_new = torch.bmm(a, enc_outputs)                                             # [batch_size, 1, enc_dim]
        context_weight_new = context_weight_new.permute(1, 0, 2)                                   # [1, batch_size, enc_dim]
        
        if self.add_extrafeat:
            rnn_input = torch.cat((rnn_input, extra_feat), dim = -1)                               # [batch_size, emb_dim + z_dim]
        if self.context_vector_input:
            rnn_input = torch.cat((rnn_input, context_weight), dim = -1)                           # [batch_size, emb_dim + dec_dim + enc_dim]
        rnn_input = rnn_input.unsqueeze(0)                                                         # [1, batch_size, emb_dim + dec_dim + enc_dim]
        rrn_out, hidden_new = self.rnn(rnn_input, hidden)                                          # [1, batch_size, dec_dim]
        
        
        #Masking individual representations padding now
        src_mask = src_mask.long()
        hidden_new = src_mask * hidden_new + (1. - src_mask) * hidden
        context_weight_new = src_mask * context_weight_new + (1. - src_mask) * context_weight
        
        hidden_new = hidden_new.squeeze(0)                                                         # [batch_size, dec_dim]
        context_weight_new = context_weight_new.squeeze(0)                                         # [batch_size, enc_dim]
        dec_out = torch.cat((hidden_new, context_weight_new, input_embs), dim = 1)                 # [batch_size, enc_dim + z_dim + num_topics + embs_dim]
        vocab_dist = torch.exp(self.out_ada.log_prob(dec_out))                                     # [batch_size, vocab_size]
        
        return vocab_dist, attn_dist, context_weight_new, hidden_new
        
class BowDecoder(nn.Module):
    """Simple Feed Forward network to decode topics into BoW vector"""
    def __init__(self, output_dim, num_topics, dropout):
        super().__init__()
        self.fc = nn.Linear(num_topics, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, qt_max):
        """
        Args:
            qt_max = [batch_size, num_topics] | Topic by document distribution
        Output:
            self.fc(qt_max) = [batch_size, vocab_size]
        """
        
        qt_max = self.drop(qt_max)                           
        return self.fc(qt_max)