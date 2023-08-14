#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import torch
import random
import json
import pandas as pd
from tqdm import tqdm
import random
import spacy
from nltk import flatten
nlp = spacy.load("en_core_web_sm")

    
def collate_tokens(values, pad_idx, left_pad=False, pad_to_length=None):

    """Convert a list of 1d tensors into a padded 2d tensor."""
    values = list(map(torch.LongTensor, values))
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

def src_mask(src, src_pad_idx):
    """Masking padding indexes for attention"""
    mask = (src != src_pad_idx)
    return mask

def flat_group(input_, leave_one_out=False):
    """Creating group representation from individual source batch of reviews"""
    batch_size = input_.shape[0]
    seq_len = input_.shape[1]
    input_ = input_.contiguous()
    flat_out = input_.view(batch_size*seq_len).unsqueeze(0).repeat(batch_size, 1)                                              # [batch_size, tot_elem]
    if leave_one_out:
        for i in range(batch_size):
            flat_out[i, seq_len * i : seq_len * i+seq_len] = 0
    
    return flat_out

def creat_tgt_ext(src_list, oovs, unk_id):
    """Initilize final target results"""
    
    #Checking how many reviews contains OOVs words - Cases
    nb_review_with_oovs = 0
    index_pos = []
    for i, oovs_list in enumerate(oovs):
        if oovs_list:
            nb_review_with_oovs += 1
            index_pos.append(i)
    
    #### Case 0 - There are no OOVs words in batch ####
    if nb_review_with_oovs == 0:
        return src_list
    
    #### Case 1 - There is only one review containing OOVs words ####
    # Then we replace those words by the unknown token
    if nb_review_with_oovs == 1:
        tgt_list = []
        for i, src in enumerate(src_list):
            tgt = []
            #The src contains no OOVs
            if i not in index_pos:
                tgt = src
            else:
                for word_id in src:
                    if word_id in oovs[i]:
                        tgt.append(unk_id)
                    else:
                        tgt.append(word_id)
            tgt_list.append(tgt)
            
        return tgt_list
                 
    #### Case 2 - There are at list two reviews with OOVs words ####
    # - we can randomly replace OOVs in Ri by an OOVs in Ri_ to ensure that the model train to select rare words from the same batch instead of hallucinating
    else:
        tgt_list = []
        for i, src in enumerate(src_list):
            tgt = []
            #The src contains no OOVs
            if i not in index_pos:
                tgt = src
            else:
                oovs_others = oovs[:i] + oovs[i+1:]
                oovs_others = flatten(oovs_others)
                intersection = list(set(oovs[i]).intersection(oovs_others))
                for word_id in src:
                    #If the OOV word is not in the other reviews we randomly sample that word to replace it for training to predict rare words - we prefer that system hallucinate a word from context that an unk
                    if word_id in oovs[i]: 
                        if word_id in intersection: 
                            tgt.append(word_id)
                        else: 
                            random_sample_oov = random.choice(oovs_others)
                            tgt.append(random_sample_oov)
                    else:
                        tgt.append(word_id)
            tgt_list.append(tgt)
        
        return tgt_list