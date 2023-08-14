#!/usr/bin/env python
# coding: utf-8
import torch
from torch.nn.functional import softmax
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from torch.distributions.log_normal import LogNormal
import numpy as np


def kld_gauss(mu_qz, sigma_qz, mu_pz, sigma_pz):
    """KL Divergence approximation for two gaussian functions - z variable"""
    posterior = Normal(loc=mu_qz, scale=sigma_qz)
    prior = Normal(loc=mu_pz, scale=sigma_pz)
    
    return kl_divergence(posterior, prior)
    

def kld_normal(mu, sigma, dim=1):
    """KL Divergence from a normal to a Gaussian assuming both have diag. covariances. - c variable"""
    return -0.5 * torch.sum(1 + torch.log(sigma) - mu.pow(2) - sigma, dim=dim)

def kld_dirichlet(mu_qt, sigma_qt, log_sigma_qt, mu_pt, sigma_pt, num_topics, dim=1):
    
    """
    KL Divergence approximation for the dirichlet distribution 
    As in: AVITM - https://github.com/estebandito22/PyTorchAVITM/blob/master/pytorchavitm/avitm/avitm.py
    """
    variance_loss = torch.sum(sigma_qt/sigma_pt, dim=dim)
    diff_mean = mu_pt - mu_qt
    deviance = torch.sum(diff_mean**2/sigma_pt, dim=dim)
    
    logvar_det_division = torch.log(sigma_pt).sum() - torch.sum(log_sigma_qt,dim=dim)

    return 0.5 * (variance_loss + deviance + logvar_det_division - num_topics)