#Main libraries
import pandas as pd
import os
from tqdm import tqdm
import nltk
import spacy
import math
import time
import sys
from nltk.corpus import stopwords
import json
import numpy as np
from numpy import dot
from numpy.linalg import norm

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gensim.models import LdaModel, CoherenceModel
from gensim.models.nmf import Nmf
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath

#
import argparse
import logging

sys.path.append("../../configs")
import config

sys.path.append("experiments/evaluation")
from ROUGE_eval import Rouge_eval, get_rouge_average, get_rouge_max
from other_evals import *

class compte_evals():
    
    def __init__(self, data_path, gen_method="NSumm"):
        
        self.data_path = data_path
        f = open(f'{data_path}')
        self.data = json.load(f)
        self.gen_method = gen_method
        
    def process_json(self, idx_):
        
        summaries = self.data[f'{idx_+1}']['summaries']
        references = self.data[f'{idx_+1}']['human_refs']
        reviews = self.data[f'{idx_+1}']['orig_reviews']
        sums, rouge_sums, refs, revs = [], [], [], []
        for j in range(len(summaries)):
            #Attention -- ROUGE function as design take list of list as input and not list of texts
            rouge_sums.append([summaries[f'gen_{j}']])
        for j in range(len(summaries)):
            sums.append(summaries[f'gen_{j}'])

        if self.gen_method=="SingleSumm":
            sums_ = '. '.join(sums)
            sums = [sums_]
            rouge_sums = [sums]

        for j in range(len(references)):
            refs.append(references[f'ref_{j}'])
        for j in range(len(reviews)):
            revs.append(reviews[f'src_{j}'])
                
        return rouge_sums, sums, refs, revs

    def compute_rouge(self, remove_stopwords=False, use_stemmer=True):

        rouge_eval = Rouge_eval(remove_stopwords=False, use_stemmer=True)
        
        for i in range(len(self.data)):
            rouge_sums, _, refs, _ = self.process_json(i)
            rouge_eval.rouge_results(references=refs, generated_summaries=rouge_sums, group_id=i+1)
        #After computing results we get avg and max scores
        rouge_results = rouge_eval.final_results
        avg_rouge_dict = get_rouge_average(rouge_results)
        max_rouge_dict = get_rouge_max(rouge_results)

        return avg_rouge_dict, max_rouge_dict

    def compute_BLEURT(self, mode="indiv"):

        bleurt_eval = Bleurt_Scoring()
        if mode == "indiv":
            bavg_indiv_revs, bmax_indiv_revs, bmin_indiv_revs =  [], [], []
            for i in range(len(self.data)):
                _, sums, _, revs = self.process_json(i)
                max_indiv_revs, avg_indiv_revs, min_indiv_revs = bleurt_eval.get_avg_bleurt(revs, sums)
                bavg_indiv_revs.append(avg_indiv_revs)
                bmax_indiv_revs.append(max_indiv_revs)
                bmin_indiv_revs.append(min_indiv_revs)
            dict_final = {"final_avg_revs":sum(bavg_indiv_revs) / len(bavg_indiv_revs),
                          "final_max_revs ":sum(bmax_indiv_revs) / len(bmax_indiv_revs),
                          "final_min_revs":sum(bmin_indiv_revs) / len(bmin_indiv_revs)}
            
        else:
            bavg_concat_revs, bmax_concat_revs, bmin_concat_revs =  [], [], []
            for i in range(len(self.data)):
                _, sums, _, revs = self.process_json(i)
                max_concat_revs, avg_concat_revs, min_concat_revs = bleurt_eval.get_avg_bleurt(revs, sums, mode='concat')
                bavg_concat_revs.append(avg_concat_revs)
                bmax_concat_revs.append(max_concat_revs)
                bmin_concat_revs.append(min_concat_revs)
            dict_final = {"final_avg_revs":sum(bavg_concat_revs) / len(bavg_concat_revs),
                          "final_max_revs":sum(bmax_concat_revs) / len(bmax_concat_revs),
                          "final_min_revs":sum(bmin_concat_revs) / len(bmin_concat_revs)}
            
        return dict_final
    
    
    def frag_coverage(self):
        """As defined in MC Clave 2001 + agrawal 2009"""
        
        avg_cover_prec, avg_cover_recall = [], []
        for i in range(len(self.data)):
            _, sums, _, revs = self.process_json(i)
            cover_prec, cover_recall = fragment_coverage(revs, sums)
            avg_cover_prec.append(cover_prec)
            avg_cover_recall.append(cover_recall)
            
        dict_final = {"frag_cover_prec": sum(avg_cover_prec) / len(avg_cover_prec),
                      "frag_cover_recall": sum(avg_cover_recall) / len(avg_cover_recall)}
        
        return dict_final
    
    def density(self):
        """As defined in MC Clave 2001 + agrawal 2009"""
        
        avg_density, max_density = [], []
        for i in range(len(self.data)):
            _, sums, _, revs = self.process_json(i)
            avg_avg, avg_max = density(revs, sums)
            avg_density.append(avg_avg)
            max_density.append(avg_max)
            
        dict_final = {"batch_avg_density": sum(avg_density) / len(avg_density),
                      "batch_max_density": sum(max_density) / len(max_density)}
        
        return dict_final

    
def diversity(pt_path):
    
    full_hidden_rep = torch.load(f'{pt_path}')
    batch_avg_sums_diversity, batch_max_sums_diversity, batch_min_sums_diversity, batch_ratio_avg, batch_ratio_max, batch_ratio_min = 0, 0, 0, 0, 0, 0
    for i, (list_hidden_sums, tensor_hidden_srcs) in enumerate(full_hidden_rep):
        
        #list_hidden_sums = [sums_size, 1, hidden_dim] / tensor_hidden_srcs = [batch_size, hidden_dim]
        tensor_hidden_sums = list_hidden_sums.squeeze(1)
        dict_div = diversity_score(tensor_hidden_srcs, tensor_hidden_sums)
        batch_avg_sums_diversity += dict_div['avg_sums_diversity'] / len(full_hidden_rep)
        batch_max_sums_diversity += dict_div['max_sums_diversity'] / len(full_hidden_rep)
        batch_min_sums_diversity += dict_div['min_sums_diversity'] / len(full_hidden_rep)
        batch_ratio_avg += dict_div['ratio_avg'] / len(full_hidden_rep)
        batch_ratio_max += dict_div['ratio_max'] / len(full_hidden_rep)
        batch_ratio_min += dict_div['ratio_min'] / len(full_hidden_rep)
        
    dict_final: {"avg_sum_div": batch_avg_sums_diversity, "max_sums_div":batch_max_sums_diversity, "min_sums_div": batch_min_sums_diversity,
                "avg_ration_div": batch_ratio_avg, "max_ration_div": batch_ratio_max, "min_ration_div": batch_ratio_min}

    
class eval_TM():
    
    def __init__(self, data_path, gen_method="NSumm", lda_path="", train_path="", review_field='', num_k=30, ban_list = ['DET', 'PUNCT', 'AUX']):
        
        self.data_path=data_path
        f = open(f'{data_path}')
        self.data = json.load(f)
        self.gen_method=gen_method
        self.train_path = train_path
        self.review_field = review_field
        self.num_k = num_k
        self.nlp = spacy.load("en_core_web_sm")
        self.ban_list = ban_list
        self.train_lda(self.train_path, self.review_field, self.num_k)
        
    #tokenizer as the one employed by the model 
    def filtered_tokenize(self, text):
        token_text = []
        doc = self.nlp(str(text))
        for token in doc:
            if not token.is_stop and token.pos_ not in self.ban_list:
                token_text.append(str(token.text))

        return token_text
    
    def process_json(self, idx_):
        
        summaries = self.data[f'{idx_+1}']['summaries']
        references = self.data[f'{idx_+1}']['human_refs']
        reviews = self.data[f'{idx_+1}']['orig_reviews']
        sums, refs, revs = [], [], []
        for j in range(len(summaries)):
            sums.append(summaries[f'gen_{j}'])

        if self.gen_method=="SingleSumm":
            sums_ = '. '.join(sums)
            sums = [sums_]

        for j in range(len(references)):
            refs.append(references[f'ref_{j}'])
        
        for i in range(len(reviews)):
            revs.append(reviews[f'src_{i}'])
        revs = ' '.join(revs)
                
        return sums, refs, revs
    
    def train_lda(self, train_path, review_field, num_k):
        df = pd.read_csv(train_path)
        reviews = list(df[review_field])
        review_token = []
        for text in reviews:
            review_token.append(self.filtered_tokenize(text))
            
        self.common_dictionary = Dictionary(review_token)
        common_corpus = [self.common_dictionary.doc2bow(text) for text in review_token]
        self.lda = LdaModel(common_corpus, id2word=self.common_dictionary, num_topics=num_k)
    
    def topic_vectorization(self, gensim_corpus):
    
        # One Third scenario - we generated only 1 summary concatenated
        full_vector_sums = []
        for unseen_sum in gensim_corpus:
            vector_ = self.lda[unseen_sum]
            topic_idx, topic_value = zip(*vector_)
            vector = []
            for i in range(self.num_k):
                if i in topic_idx:
                    idx = topic_idx.index(i)
                    vector.append(topic_value[idx])
                else:
                    vector.append(0.)
            full_vector_sums.append(vector)

        return full_vector_sums
    
    def compute(self):
        orig_batch, gen_sums, all_refs= [], [], []
        for i in range(len(self.data)):
            sums, refs, revs = self.process_json(i)
            gen_sums.append(sums)
            all_refs.append(refs)
            orig_batch.append(revs)
        
        batch_token = []
        for text in orig_batch:
            batch_token.append(self.filtered_tokenize(text))
        if self.gen_method == "NSumm":
            sum_token = []
            for list_sums in gen_sums:
                for text in list_sums:
                    sum_token.append(self.filtered_tokenize(text))
        else:
            sum_token = []
            for text in gen_sums:
                sum_token.append(self.filtered_tokenize(text))
        
        refs_token = []
        for list_refs in all_refs:
            for text in list_refs:
                refs_token.append(self.filtered_tokenize(text))
        
        batch_corpus = [self.common_dictionary.doc2bow(text) for text in batch_token]
        sum_corpus = [self.common_dictionary.doc2bow(text) for text in sum_token]
        refs_corpus = [self.common_dictionary.doc2bow(text) for text in refs_token]
        
        vector_batch_reviews = self.topic_vectorization(batch_corpus)
        vector_sums = self.topic_vectorization(sum_corpus)
        vector_refs = self.topic_vectorization(refs_corpus)
        
        return vector_sums, vector_batch_reviews, vector_refs
    
    
    # Best matching topic & vocabulary proximity
    def term_coverage(self, refs, sums):
        count = 0.
        for word_idx, word in enumerate(refs):
            if word in sums:
                sum_idx = sums.index(word)
                score = 1/(word_idx+1) * 1/(sum_idx+1)
                count += score
        return count

    def best_topic(self, sum_vector, ref_vector, top=100, n_gens=3):

        commonW_list = []
        count_best_match = 0

        for i in range(len(ref_vector)):
            ref_best_topic = np.argmax(ref_vector[i])
            if self.gen_method == "NSumm":
                assert len(sum_vector) == n_gens*len(ref_vector)
                idx_ = i*n_gens
                sum_best_topics= []
                for j in range(n_gens):
                    sum_best_topics.append(np.argmax(sum_vector[idx_+j]))
                
                ref_terms = self.lda.get_topic_terms(ref_best_topic, topn=top)
                word_ref_idx, word_ref_value = zip(*ref_terms)
                score_cover_list = []
                for sum_best_topic in sum_best_topics:
                    #Counting number of times where best topic are the same
                    if sum_best_topic == ref_best_topic:
                        count_best_match += 1
                    #Estimating how many words in common between topics even if they are not the same
                    sum_terms = self.lda.get_topic_terms(sum_best_topic, topn=top)
                    word_sum_idx, word_sum_value = zip(*sum_terms)
                    score_cover = self.term_coverage(word_ref_idx, word_sum_idx)
                    score_cover_list.append(score_cover)
                score_cover = max(score_cover_list)
                commonW_list.append(score_cover)

            else:
                assert len(sum_vector) == len(ref_vector)
                sum_best_topic = np.argmax(sum_vector[i])
                ref_terms = self.lda.get_topic_terms(ref_best_topic, topn=top)
                word_ref_idx, word_ref_value = zip(*ref_terms)
                #best topics
                if sum_best_topic == ref_best_topic:
                    count_best_match += 1

                # matching keywords between topics
                sum_terms = self.lda.get_topic_terms(sum_best_topic, topn=top)
                word_sum_idx, word_sum_value = zip(*sum_terms)
                score_cover = self.term_coverage(word_ref_idx, word_sum_idx)
                commonW_list.append(score_cover)

        average_sim = sum(commonW_list) / len(commonW_list)

        return count_best_match, average_sim, commonW_list
    
    def cosine_sim(self, a, b):
        return dot(a, b)/(norm(a)*norm(b))

    def sim_topics(self, sum_vector, ref_vector, n_gens=3):

        if self.gen_method == "NSumm":
            assert len(sum_vector) == 3*len(ref_vector)
            cossim_list = []

            for i in range(len(ref_vector)):
                idx_ = i*n_gens
                temp_cos = []
                for j in range(n_gens):
                    temp_cos.append(self.cosine_sim(sum_vector[idx_+j], ref_vector[i]))
                cossim = max(temp_cos)
                cossim_list.append(cossim)
        else:
            assert len(sum_vector) == len(ref_vector)
            cossim_list = []
            for i in range(len(sum_vector)):
                cossim_list.append(self.cosine_sim(sum_vector[i], ref_vector[i]))

        average_sim = sum(cossim_list) / len(cossim_list)
        
        return average_sim
    
if __name__ == "__main__":
    
    ##define args here
    results_path = "../results/results_Config9_TopicsGen=6_K8_Normal.json"
    _pt_path = "../results/hidden_Config9_TopicsGen=6_K8_Normal.pt"
    train_data = "../../data/20220510_amazon_reviews_train_processed.csv"
    save_results_path = "experiments/results/results_Config9_TopicsGen=6_K8_Normal.csv"
    
    #IF conditions - depending on the choice in config
    #compute = compte_evals(results_path, gen_method="NSumm")
   # avg_rouge_dict, max_rouge_dict = compute.compute_rouge()
    #filt_avg_rouge_dict, filt_max_rouge_dict = compute.compute_rouge(remove_stopwords=True)
    #bleurt_dict = compute.compute_BLEURT()
   # frag_dict = compute.frag_coverage()
   # dens_dict = compute.density
   # diversity_ = diversity(_pt_path)
    
    compute_TM = eval_TM(results_path, gen_method="NSumm", train_path=train_data, review_field='review', num_k=30, ban_list = ['DET', 'PUNCT', 'AUX'])
    vector_sums, vector_batch_reviews, _ = compute_TM.compute()
    count_best_match, average_sim_overlap, _ = compute_TM.best_topic(vector_sums, vector_batch_reviews, top=100, n_gens=3)
    avg_topic_sim = compute_TM.sim_topics(vector_sums, vector_batch_reviews, n_gens=3)
    
    #Papers indicated metrics
    dict_final = {"R1-avg": avg_rouge_dict["avg_max_1"], "R1-max":max_rouge_dict["max_max_1"],
                  "R2-avg": avg_rouge_dict["avg_max_2"], "R2-max":max_rouge_dict["max_max_2"],
                  "RL-avg": avg_rouge_dict["avg_max_L"], "RL-max":max_rouge_dict["max_max_L"],
                  "R1-filt-avg": filt_avg_rouge_dict["avg_max_1"], "R1-filt-max":filt_max_rouge_dict["max_max_1"],
                  "R2-filt-avg": filt_avg_rouge_dict["avg_max_2"], "R2-filt-max":filt_max_rouge_dict["max_max_2"],
                  "RL-filt-avg": filt_avg_rouge_dict["avg_max_L"], "RL-filt-max":filt_max_rouge_dict["max_max_L"],
                  "BELURT": bleurt_dict["final_avg_revs"], "Word_Overlap": average_sim_overlap, "Topic_similarity":avg_topic_sim}
    
    #save in pandas csv
    df = pd.DataFrame(dict_final, index=[0])
    df.to_csv(save_results_path, index=False)