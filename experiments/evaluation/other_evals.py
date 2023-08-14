#!/usr/bin/env python
# coding: utf-8

#Main libraries
import os
import nltk
import spacy
import math
from nltk.corpus import stopwords
import numpy as np

#torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")

# BLEURT - avec Ref et avec original documents en MDS ou en concat'
from bleurt import score


class Bleurt_Scoring():
    
    """
    HOW TO INTEPRET : The BLEU SCORE IS A METRIC TO CORRELATE HUMAN JUDGEMENT NOT TO INTERPRET THE VALUE PER SE
                    The more the value tend to negative the less probable a human is to judge similar output for the two text
                    Used to compare to systems : if average of one is higher than average of the second then first is more "similar" to the second that's it
    """
    
    def __init__(self):
        
        self.checkpoint = "/home/florian_carichon_rdccaa_com//bleurt/bleurt/test_checkpoint"
        self.scorer = score.BleurtScorer(self.checkpoint)
        
    def bleurt_indiv_score(self, src, sum_):
        
        """
        Return BLEURT AVG, MIN, MAX scores of bleurt of 1 summary with all references
        Args:
            src: references = type(list)
            sum_: Generated summaries = type(text)
        Output:
            max_score, avg_score, min_score : type(float) - BLEURT SCORES
        """
        total_scores = []
        for text in src:
            scores = self.scorer.score(references=[text], candidates=[sum_])
            total_scores += scores

        max_score = max(total_scores)
        avg_score = sum(total_scores) / len(total_scores)
        min_score = min(total_scores)
        
        return max_score, avg_score, min_score
            
    def bleurt_concat_score(self, src, sum_):

        """
        Return BLEURT AVG, MIN, MAX scores of bleurt of 1 summary with one reference (concatenated version of the 3 references)
        Args:
            src: references = type(list)
            sum_: Generated summaries = type(text)
        Output:
            score[0] : type(float) - BLEURT SCORE
        """
        
        src_concat = ' '.join(src)
        score = self.scorer.score(references=[src_concat], candidates=[sum_])
            
        return score[0]
    
    def get_avg_bleurt(self, srcs, sums, mode='indiv'):
        """Get average score for all batches
        Args:
            srcs: list of all references = type(list)
            sum_: List of all generated summaries = type(list)
        Output:
            Average max_score, avg_score, min_score - BLEURT SCORES
        """
        if mode == 'indiv':
            max_sum_score = []
            avg_sum_score = []
            min_sum_score = []
            for summary in sums:
                max_, avg_, min_ = self.bleurt_indiv_score(srcs, summary)
                max_sum_score.append(max_)
                avg_sum_score.append(avg_)
                min_sum_score.append(min_)
        
            return sum(max_sum_score) / len(max_sum_score), sum(avg_sum_score) / len(avg_sum_score), sum(min_sum_score) / len(min_sum_score)
        
        elif mode == 'concat':
            sum_score = []
            for summary in sums:
                score = self.bleurt_concat_score(srcs, summary)
                sum_score.append(score)
            
            return max(sum_score), sum(sum_score) / len(sum_score), min(sum_score)
            
        else:
            raise Error ('please entern valid value indiv or concat for mode')


def diversity_calulation(text):
    
    """
    HOW to interpret and how to use : 
        Measure the cosine similarity between two representation.
        If the summaries proposed by humans are diverse and if input is diverse then output should be diverse too. 
        Mesurer les ratio de diversité entre les outputs systems et les inputs pour interprétation
    
    # Mc Clave 2001 version of Diversity : Sum distance of all texts / nb_text
    Input : text type(tensor[bs,dim]) - hidden_BOW representation of input srcs
    Output : div_score type(float) - Absolute diversity between summaries
             reative_div_score type(float) - Diversity ration between src and summaries
    """
    
    tot_avg_sim = []
    tot_max_sim = []
    tot_min_sim = []
    cos = torch.nn.CosineSimilarity(dim=0)
    
    for i, review in enumerate(text):
        other_reviews = torch.cat((text[:i], text[i+1:]))
        score_by_review = []
        for other_review in other_reviews:
            cosine_dist = cos(review, other_review)
            score_by_review.append(cosine_dist)
        
        #We compute avg score of individual review with all other / max score of each review / min score of each review - results = [batch_size]
        tot_avg_sim.append(sum(score_by_review) / len(score_by_review))
        tot_max_sim.append(max(score_by_review))
        tot_min_sim.append(min(score_by_review))
    
    #Averaging the score over the batch to know the average distance of all reviews to each others
    avg_sim = sum(tot_avg_sim) / len(tot_avg_sim)
    max_sim = sum(tot_max_sim) / len(tot_max_sim)
    min_sim = sum(tot_min_sim) / len(tot_min_sim)
    
    return avg_sim, max_sim, min_sim


def diversity_score(src, sums):
    
    """
    # Mc Clave 2001 version of Diversity : Sum distance of all texts / nb_text
    Input : src type(tensor[bs,dim]) - hidden_BOW representation of input srcs
            sums type(tensor[bs,dim]) - hidden_BOW representation of input summaries
    Output : div_score type(float) - Absolute diversity between summaries
             reative_div_score type(float) - Diversity ration between src and summaries
    """
    
    avg_src_diversity, max_src_diversity, min_src_diversity = diversity_calulation(src)
    avg_sums_diversity, max_sums_diversity, min_sums_diversity = diversity_calulation(sums)
    
    dict_result = {'avg_src_diversity': avg_src_diversity.item(),
                   'max_src_diversity': max_src_diversity.item(),
                   'min_src_diversity': min_src_diversity.item(),
                   'avg_sums_diversity': avg_sums_diversity.item(),
                   'max_sums_diversity': max_sums_diversity.item(),
                   'min_sums_diversity': min_sums_diversity.item(),
                  'ratio_avg': avg_src_diversity/avg_sums_diversity.item(),
                  'ratio_max': max_src_diversity/max_sums_diversity.item(),
                  'ratio_min': min_src_diversity/min_sums_diversity.item()}
    
    return dict_result

def tokenize(text):
    nlp = spacy.load("en_core_web_sm")
    return [str(word) for word in nlp(str(text)) if word.text not in stopwords.words('english')]

# Grusky 2018 
def fragment_coverage(src, sums):
    """
    Objective here is to demonstrate that the concatenation of list of sums in our case cover more of the input than 3 times the coverage proposed by bravzinkas (normalization needed!)
    src : type(list(texts))
    sums = type(list(text))
    return type(float)
    no_stemmer (à faire?)
    """

    src_all = ' '.join(src)
    sums_all = ' '.join(sums)

    src_all_tokens = tokenize(src_all)
    sums_all_tokens = tokenize(sums_all)
    
    count_in_ref = 0
    for word in sums_all_tokens:
        if word in src_all_tokens:
            count_in_ref += 1

    fragment_cov_precision = count_in_ref / len(sums_all_tokens)
    fragment_cov_recall = count_in_ref / len(src_all_tokens)
    
    return fragment_cov_precision, fragment_cov_recall


# Grusky 2018 
def density(srcs, sums):
    
    nlp = spacy.load("en_core_web_sm")
    avg_avg_sum = []
    avg_max_sum = []
    for summary in sums:
        #Enelver les stopwords ici?
        sum_tokens = tokenize(summary)
        ref_sum_len = []
        for src in srcs:
            src_tokens = tokenize(src)
            ref_sum_len.append(indiv_density(src_tokens, sum_tokens))
        
        avg_avg_sum.append(sum(ref_sum_len) / len(ref_sum_len))
        avg_max_sum.append(max(ref_sum_len))
        
    avg_avg = sum(avg_avg_sum) / len(avg_avg_sum)
    avg_max = sum(avg_max_sum) / len(avg_max_sum)
    
    return avg_avg, avg_max


def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list

def indiv_density(src,sum_):
    
    """
    Input : src : token list of one src reference
            sum_ : token list of one summary
    """
    sum_length = []
    for i, word in enumerate(sum_):
        index_length_list = []
        get_potential_index = get_index_positions(src, word)
        #check potential presence of word in list
        if len(get_potential_index) > 0:
            word_length = 1
        else: 
            continue
        for index in get_potential_index:
            previous, following = True ,True
            previous_counter = 1
            following_counter = 1
            #We check all previous extractive members
            while previous:
                if index-previous_counter < 0 or i-previous_counter <0:
                    previous = False
                else:
                    if src[index-previous_counter] == sum_[i-previous_counter]:
                        previous_counter += 1
                        word_length += 1
                    else:
                        previous = False
            while following:
                if index+following_counter == len(src) or i+following_counter == len(sum_):
                    following = False
                else:
                    if src[index+following_counter] == sum_[i+following_counter]:
                        following_counter += 1
                        word_length += 1
                    else:
                        following = False
            
            index_length_list.append(word_length)
            
            tot_extractive_word_length = max(index_length_list)
        
        sum_length.append(tot_extractive_word_length)
    
    #If not a single word is shared between the two texts
    if len(sum_length) == 0:
        sum_length.append(0)
        
    avg_extract_length = sum(sum_length)/len(sum_length)
    
    return avg_extract_length