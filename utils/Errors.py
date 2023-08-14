#!/usr/bin/env python
# coding: utf-8

from utils.model_utils import SeqToBow
import numpy as np
import torch.nn as nn
import torch

def loss_estimation(output, avg_c_kl, avg_z_kl, avg_t_kl, tgt_ext, vocab, BoW_src, Bow_outputs, cycle, annealing_t, z_weight_max=1.0, c_weight_max=0.6, t_weight_max=1.0):
    """
    Args:
        output: Model final distribution predicted
        avg_c_kl, avg_z_kl, avg_t_kl: kl divergence for respectively c, z, and t
        tgt_ext: Target to predict == input in our case of self-supervised
        vocab: vocabulary associated to the model
        BoW_src: Target BoW vectors
        Bow_outputs: Predicted BOW distribution
        cycle: Annealing cycle for our latent parameters
        annealing_t: Annealing linear ration for the topics
        z_weight_max, c_weight_max, t_weight_max: ponderation for accounting for divergence -- beta VAE
        epoch : current epoch of training
        iter_ : i-th element in iterator 
        epoch_size : number of steps in training for update KL weight iteratively depending on steps
    Output:
        rec_loss: General error
        kl_loss, BoW_loss, kld_c, kld_z, kld_t: Individual losses -- keep for tracking through tensorboard
    """
    # Criterions for losses : 
    criterion = nn.NLLLoss(ignore_index=vocab.pad(), reduction='none')
    criterion_BoW = nn.MSELoss()
    
    #Negative log likelihood loss for reconstruction
    output = torch.log(output)
    rec_loss = criterion(output, tgt_ext)
    rec_loss = torch.mean(rec_loss)
    
    #KL divergence loss
    kl_weight_z = cycle * z_weight_max
    kl_weight_c = cycle * c_weight_max
    kl_weight_t =  annealing_t * cycle * t_weight_max
    kld_c = avg_c_kl * kl_weight_c
    kld_z = avg_z_kl * kl_weight_z
    kld_t = avg_t_kl * kl_weight_t
    
    kl_loss = kld_c + kld_z + kld_t
    
    #Bag Of Words normalization loss
    if Bow_outputs == None:
        BoW_loss = 0
    else:        
        BoW_loss = torch.mean(-torch.sum(BoW_src * torch.log(Bow_outputs + 1e-10), dim=1))

    return rec_loss, kl_loss, BoW_loss, kld_c, kld_z, kld_t