#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT_DIR = "/home/florian_carichon_rdccaa_com/TopicDiversifiedVAESumm/"

path = {
    "root": ROOT_DIR,
    "data": f"{ROOT_DIR}data/",
    "tota_vocab" : f"{ROOT_DIR}data/totalrevs&bravrevs_extvocab.csv",
    "extended_vocab" : f"{ROOT_DIR}data/ext_vocab_full.pkl",
    "vocab_path": f"{ROOT_DIR}data/vocab.pkl",
    "train": f"{ROOT_DIR}data/20220510_amazon_reviews_train_processed.csv",
    "valid": f"{ROOT_DIR}data/20220510_amazon_reviews_valid_processed.csv",
    "test": f"{ROOT_DIR}data/final_evalbravinska_ROUGE.csv"
}

data = {
    "vocab_size" : 20000,
    "ext_vocab_size" : 26000,
    "min_freq" : 1,
    "max_num_reviews": 8,
    "max_len_rev" : 80,
    "shuffle_revs": False,
    "n_prod_shuf": 3,
    "n_fus": 0,                                    #Set to 2 for 16 reviews per batch
    "recons_test": True,
    "shuffle_batch": False,
    "ban_list": ['DET', 'PUNCT', 'AUX'],
    "raw_data": False,
    "topic_encod" : 'INDIV',                       #Type of BoW representations
    "encode_type":'BoW',
    "review_field":"review"
    }

###ATTENTION -- TROUVER UN MOYEN DE METTRE LAVARIANCE EN FONCTION DU NOMBRE DE TOPIC ET PAS EN BRUT A 30
model = {
    "emb_dim" : 200,
    "enc_dim" : 512,
    "dec_dim" : 512,
    "attn_dim" : 300,
    "c_dim" : 600,
    "z_dim" : 600,
    "num_topics" : 30,
    "drop" : 0.2,
    "use_pretrained" : True,
    "n_layers" : 1 ,    #Use for both encoder and decoder
    "gen_cond" : 'X',
    "topic_prior" : "Uniform",
    "BoW_train" : True,
    "Glove_name" : '6B',
    "encod_bidir":True,
    "topic_prior_mean":0.0,
    "topic_prior_variance": 1.-(1./30),
    "add_extrafeat":False,
    "add_extrafeat_pgn": False,
    "context_vector_input":True,
    "use_topic_attention":False,
    }

experiments = {
    "writer" : True,
    "Name" : "default_config",
    "write_rec_loss" : True,
    "write_kl_loss" : True,
    "write_BoW_loss" : True,
    "write_total_loss" : True,
    "generate":False,
    "compute_evals":False,
    "save_name": "default_config",                   # Don't put file extension here - just the config name -- used for different multiple saves
    "remove_stopwords" : False,
    "use_stemmer" : False,
    "train_model": True                             # I honestly don't understand why the value of this one can't be change with agr parser....
    }

train_params = {
    "optimizer" : torch.optim.Adam,
    "clip" : 10.0,
    "lr" : 5e-4,
    "weight_decay" : 1e-6,
    "accumulation_steps" : 32,
    "tf":0.85,                      #teacher_forcing_ratio
    "momentum": 0.99,
    "N_epoch": 200,
    "z_weight_max" : 1.0,
    "c_weight_max" : 0.65,
    "t_weight_max" : 0.65,
    "n_epoch_cycle": 8,
    "cycle_iter":0,
    "n_cycles_ct":1,
    "start_lin_t":0,
    "duration_lin_t":40
    }

inference = {
    "beam_size" : 5,
    "min_steps" : 30,
    "num_return_seq" : 5,
    "num_return_sum" : 1,
    "n_gram_block": 3,
    "gen_len_method": "NSumm",   #SingleSumm
    "topic_gen": "select_main", #word_bias
    "gen_word": [], 
    "num_t_gen":3,
    "t_mask_len":8,
    "rencode":True
    }