#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import re
import torch.distributions.dirichlet as d
import torch.nn.functional as F
from utils.model_utils import filter_oov
import random
import spacy

def postprocess(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True):
    """Removing special tokens and spaces in generation"""
    if skip_special_tokens:
        tokens = [t for t in tokens if not is_special(t)]
    out_string = ' '.join(tokens)
    
    if clean_up_tokenization_spaces:
        out_string = clean_up_tokenization(out_string)
    
    return out_string

def is_special(token):
    res = re.search("\<[a-z]+\>", token)
    if res is None:
        return False
    return token == res.group()


def clean_up_tokenization(out_string):
    """
    Reference : transformers.tokenization_utils_base
    Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
    Args:
        out_string (:obj:`str`): The text to clean up.
    Returns:
        :obj:`str`: The cleaned-up string.
    """
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string
    
def ngramm_mirror(current_summary, topk, vocab, mirror_conjs = ["and", 'or', ',', 'but', '.'], gram_mirror_window=2):
    
    """
    Preventing indirect token repetition in mirror configurations especially around special tokens
    Generalize later - for now we only manage mirror windows of size 2
    
    Args: 
        current_summary: Generated text -- id list
        topk : current potential prediction | should have a size of 3 potential solutions max - we don't consider farther away in this case
        vocab: vocabulary -- matching id list to words
        mirror_conjs: special token liost where we watch mirrors
        gram_mirror_window: size around the special token to watch
    Output:
        topk -- filtered problematic token for mirroring
    """
    current_summary = torch.stack(current_summary, dim=-1)
    
    #Condition for mirroring - length of current summary must be greater than 3 tokens
    temp = []
    for conj in mirror_conjs:
        temp.append(vocab.word2id(conj))
    
    mirror_idxs = torch.tensor(temp)
    summaries_prev = current_summary[:,-gram_mirror_window:]
    
    #summaries_prev_to_check = topic_summary[:,-gram_mirror_window-1:]
    summaries_prev_to_check = current_summary[:,-gram_mirror_window-2:]
    
    #Identifying rows and columns in summary that include an index in the last mirror_idxs position
    mirror_present = sum(summaries_prev==i for i in mirror_idxs).bool()
    row, column = (mirror_present == True).nonzero(as_tuple=True)
    
    #Case 1 - [a,b, AND, b, ...] - we prevent a direct mirror
    #Getting rows and column associated when the mirror tokens was generated previous iteration
    column_last_gen = (column == 1).nonzero(as_tuple=True)[0]
    row_last_gen = row[column_last_gen]
    
    #Getting the values from topk and current summary to check if mirroring appears
    #summaries_prev_to_check[row_last_gen, -1] = mirror index in that case
    sum_to_change = summaries_prev_to_check[row_last_gen, -2]
    topk_to_change = topk[row_last_gen,0]
    #Identifying rows where the mirror actually appears
    row_to_change = (sum_to_change == topk_to_change).nonzero(as_tuple=True)[0]
    
    if row_to_change.shape[0] > 0 :
        #Swapping position of the two first column of 
        #Exception to create here - if [a,b, AND, b, a, ...], we don't want to swap b and a but b and c (3rd in list)
        if (summaries_prev_to_check[row_last_gen, -3] == topk[row_last_gen, 1]).nonzero(as_tuple=True)[0].shape[0] > 0:
            topk[row_to_change] = torch.index_select(topk[row_to_change], 1, torch.LongTensor([2,1,0]).to(topk.device))
        else:
            topk[row_to_change] = torch.index_select(topk[row_to_change], 1, torch.LongTensor([1,0,2]).to(topk.device))
    
    #Case 2 - [a,b, AND, a, b, ...] - we prevent a second degree mirror
    #Getting rows and column associated when the mirror tokens was generated two iterations ago
    #current summary structure is [a,b,mirror, a] and topk potential b
    column_lasttwo_gen = (column == 0).nonzero(as_tuple=True)[0]
    row_lasttwo_gen = row[column_lasttwo_gen]
    
    #Getting the values from topk and current summary to check if mirroring appears
    sum_to_check = current_summary[row_lasttwo_gen]
    topk_to_check = topk[row_lasttwo_gen]
    
    #Creating the potential mirror 
    mirror = torch.concat((sum_to_check[:,-1].unsqueeze(-1), topk_to_check[:,0].unsqueeze(-1)), dim=1)
    #Checking if the mirror exists and getting the associated rows
    mirror_presence = mirror == sum_to_check[:,-gram_mirror_window-2:-gram_mirror_window]
    true_mirror = torch.all(mirror_presence, dim=1)
    row_to_change = row_lasttwo_gen[true_mirror]
    
    if row_to_change.shape[0] > 0 : 
            topk[row_to_change] = torch.index_select(topk[row_to_change], 1, torch.LongTensor([1,0,2]).to(topk.device))
    
    return topk