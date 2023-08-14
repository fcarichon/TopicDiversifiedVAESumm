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
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class mu_sigma_qc(nn.Module):
    """Sampling variable z from group encodings"""
    def __init__(self, group_dim, c_dim):
        super().__init__()
        
        self.group_dim = group_dim
        self.c_dim = c_dim
        self.mu_sigma = nn.Linear(group_dim, 2 * c_dim)
        
    def forward(self, group_input):
        """
        Arg: group_input = [batch_size, total_els_group] | Group_encoding with attention
        Output: Mu and sigma posterior parameters for sampling c through Gaussian
        """
        output = torch.tanh(self.mu_sigma(group_input))
        mu = output[:, :self.c_dim]                                  #[batch_size, c_dim]
        log_sigma = output[:, self.c_dim:]                           #[batch_size, c_dim]
        sigma = torch.exp(log_sigma)
        
        return mu, sigma


class mu_sigma_qz(nn.Module):
    """Sampling posterior param for variable z from individual reviews and c"""
    def __init__(self, dec_dim, c_dim, z_dim):
        super().__init__()

        input_dim = dec_dim + c_dim
        self.z_dim = z_dim
        self.mu_sigma = nn.Linear(input_dim, 2 * z_dim)
        self.bn_z = nn.BatchNorm1d(2 * z_dim)
        
    def forward(self, group_input):
        """
        Arg: group_input = [batch_size, dec_dim + c_dim] | Concat of text hidden and c
        Output: Mu and sigma posterior parameters for sampling z
        """
        output = torch.tanh(self.bn_z(self.mu_sigma(group_input)))    #[batch_size, 2*z_dim]
        mu = output[:, :self.z_dim]                                   #[batch_size, c_dim]
        log_sigma = output[:, self.z_dim:]                            #[batch_size, c_dim]
        sigma = torch.exp(log_sigma)
        
        return mu, sigma
    
class mu_sigma_pz(nn.Module):
    """Sampling prior parameters of variable z from c only"""
    def __init__(self, c_dim, z_dim):
        super().__init__()

        self.c_dim = c_dim
        self.z_dim = z_dim
        self.mu_sigma = nn.Linear(c_dim, 2*z_dim)

    def forward(self, c):
        """
        Arg: group_input = [batch_size, c_dim]
        Output: Mu and sigma prior parameters for sampling z
        """
        output = torch.tanh(self.mu_sigma(c))                        # [batch_size, 2*z_dim]
        mu = output[:, :self.z_dim]                                  # [batch_size, c_dim]
        log_sigma = output[:, self.z_dim:]                           # [batch_size, c_dim]
        sigma = torch.exp(log_sigma)
        
        return mu, sigma
    
class mu_sigma_qt(nn.Module):
    """Sampling variable t from Bag of words representation"""
    def __init__(self, bow_dim, num_topics):
        super().__init__()
        #8 Is chosen just to compress intermediate representation to gether more complex relations - Can be changed - no real influence on end performace as long as you keep two layers
        self.bow_dim = bow_dim
        self.num_topics = num_topics
        self.lin_inter = nn.Linear(bow_dim, 8 * num_topics)
        self.activation = nn.ReLU()
        self.mu_sigma = nn.Linear(8 * num_topics, 2 * num_topics)
        
    def forward(self, bow_input):
        """
        Arg: bow_input = [batch_size, dec_dim]  | BoW hidden representation
        Output: Mu and sigma posterior parameters for sampling t
        """              
        inter_ = self.activation(self.lin_inter(bow_input))           # [batch_size, 8*num_topics]
        output = torch.tanh(self.mu_sigma(inter_))                    # [8*num_topics, 2*num_topics]
        mu = output[:, :self.num_topics]                              # [batch_size, num_topics]
        log_sigma = output[:, self.num_topics:]                       # [batch_size, num_topics]
        sigma = torch.exp(log_sigma)
        
        return mu, sigma, log_sigma