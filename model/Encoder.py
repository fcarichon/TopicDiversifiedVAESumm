#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class GRU_Encoder(nn.Module):
    """Encoding sentences for LM model with GRU"""
    def __init__(self, input_dim, emb_dim, enc_dim, dec_dim,dropout, bidir=True):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim, enc_dim, num_layers=1, dropout=dropout, bidirectional=bidir, batch_first=True)
        if bidir:
            self.fc = nn.Linear(enc_dim * 2, dec_dim) 
        else:
            self.fc = nn.Linear(enc_dim, dec_dim)

    def forward(self, embeddings, src_max_len, rencode=False):
        """
        Args:
            src = [src_len, batch_size]
            embedded = [src_len, batch_size, emb_dim]
            src_max_len : [batch_size]
            rencode: re-encoding summary sentences to measure diversity
        Output:
            outputs: All sequence representations
            hidden: Sentence representation
        """
        #Managing packed sentences
        if rencode == False:
            packed_embedded = nn.utils.rnn.pack_padded_sequence(embeddings, src_max_len.to('cpu'), enforce_sorted=False)
            packed_outputs, hidden = self.rnn(packed_embedded)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        
        #In re-encoding process where we don't need packed sentence since we have only batch_size_dim=1 input
        else:
            embeddings_ = embeddings.permute(1,0, 2)
            outputs, hidden = self.rnn(embeddings_)
        
        outputs = self.dropout(outputs)                                                                    #[seq_len, batch_size, enc_dim*2]
        hidden = self.dropout(hidden)                                                                      #[num_layers, batch_size, enc_dim]
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))                 #[batch size, dec_dim]
        
        return outputs, hidden

    
class BoWEncoder(nn.Module):
    """Encoding sentences for BoW/Topic Model"""
    def __init__(self, input_dim, enc_dim, dec_dim, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.enc_dim = enc_dim
        self.enc_dim = dec_dim
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, enc_dim)
        self.fc2 = nn.Linear(enc_dim, dec_dim)
        self.softplus = nn.Softplus()
        
    def forward(self, embeddings_BoW):
        """
        Args:
            embeddings_BoW = [batch_size, vocab_size]
        Output:
            hidden_BoW: Sentence representation
        """
        hidden_BoW = self.softplus(self.fc1(embeddings_BoW))
        hidden_BoW = self.softplus(self.fc2(hidden_BoW))
        hidden_BoW = self.dropout(hidden_BoW)                   #[batch_size, dec_dim]
        
        return hidden_BoW