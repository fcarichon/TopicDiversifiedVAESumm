#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import GloVe
use_gpu = torch.cuda.is_available()
import numpy as np
device = torch.device("cuda:0" if use_gpu else "cpu")
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def load_gloveembeddings(vocab, Glove_name='6B', Glove_dim=100):
    """Loading Embedding representation
    Args:
        vocab: Vocabulary words for generating embeddings
        Glove_name: Glove Name pretrained model
        Glove_dim: dimesion of the model
    """
    
    emb_vecs = []
    glove_vecs = GloVe(name=Glove_name, dim=Glove_dim, unk_init=torch.Tensor.normal_)
    for w in vocab._id_to_word:
        w_emb = glove_vecs[w]
        emb_vecs.append(w_emb)
    emb_vecs = list(map(lambda x: x.squeeze(), emb_vecs))
    emb_vecs = torch.stack(emb_vecs)

    return emb_vecs

class SeqToBow(nn.Module):
    """Transform sequence into count on vocab dim for each element of batch"""
    def __init__(self, vocab, topic_encod='GROUP', encode_type='BoW'):
        """
        Args:
            vocab: Vocabulary
            topic_encod: GROUP - create the same representation for all review in the batch as agglomerated representation | INDIV: each review has its BoW representation
            encode_type: BoW - Classic Bag of Word with frequency of terms | one_hot - Binary indicating presence of term in reviews
        """
        super().__init__()
        self.vocab = vocab
        self.input_dim = len(vocab)
        self.topic_encod = topic_encod
        self.encode_type = encode_type

    def forward(self, src, ignore_index):
        """
        Args:
            src = [seq_len, batch_size]
            ignore_index = indices to ignore -- padding index
        Output:
            counts: [batch_size, vocab_size] Final BoW representation for the batch
        """        
        batch_size = src.shape[1]
        
        if self.encode_type == 'one_hot':
            counts = torch.zeros((src.size(0), self.input_dim), dtype=torch.float, device=src.device)
            ones = torch.ones_like(src, dtype=torch.float,device=src.device)
            counts.scatter_(1, src, ones)
            counts[:, ignore_index] = 0
            
            #If we want to model topic distribution of whole group information
            if self.topic_encod == 'GROUP':
                #collecting the 1 values n each dimension
                counts = counts.max(dim=0).values
                counts = counts.unsqueeze(0).repeat(batch_size,1)

        else: 
            src = src.permute(1,0)
            counts = torch.zeros((src.size(0), self.input_dim), dtype=torch.float, device=src.device)
            ones = torch.ones_like(src, dtype=torch.float,device=src.device)
            counts.scatter_add_(1, src, ones)
            counts[:, ignore_index] = 0
            
            #If we want to model topic distribution of whole group information
            if self.topic_encod == 'GROUP':
                #as BoW, the representation for whole group is the sum of each elem of batch
                counts = torch.sum(counts, dim=0)                                            # [vocab_size]
                #Creating the same encoding  for each element of the same group
                counts = counts.unsqueeze(0).repeat(batch_size,1)                            #[batch_size, vocab_size]
        
        #Putting 0 value for special tokens eos and sos
        counts[:,self.vocab.word2id('<sos>')] = 0
        counts[:,self.vocab.word2id('<eos>')] = 0
        
        return counts 
    
def filter_oov(tensor, vocab):
        """ Replace any OOV index in `tensor` with <unk> token """ 
        result = tensor.clone()
        result[tensor >= len(vocab)] = vocab._word_to_id["<unk>"]
        return result
    

def get_MMR_dist(post_dist, prior_dist, n=3, lambda_=0.2, sim_mesure="cosine"):
    """
    Measuring distance from topic distribution and its prior then selecting best topics with a MMR function
    Args:
        post_dist: Posterior distribution
        prior_dist: Prior distribution
        n: Number of topics to get back -- number of indices
        lambda_: MMR model promoting diversity vs topic relevance
        sim_mesure: cosine or kl_divergence | Which distance used to measure the distance between prior and posterior
    Output:
        doc_to_select: Indices of best topics to select - the one most divergent to their prior distribution
    """
    
    post_dist = post_dist.cpu()
    prior_dist = prior_dist.cpu()
    if sim_mesure=="cosine":
        sim = F.cosine_similarity(post_dist, prior_dist, dim=1)                                    # [num_topics]
    else: 
        sim = F.kl_div((post_dist).log(), prior_dist, None, None, reduction='none').mean(dim=1)    # [num_topics]
    doc_to_select = []
    idx = np.array(range(sim.shape[0]))
    
    #ATTENTION - DIFFERENT IMPLEMENTATION FROM COSINE Which is a distance and kl which is a similarity metric
    #Instantiating the MMR part 
    while len(doc_to_select) < n:
        if len(doc_to_select) == 0:
            if sim_mesure == "cosine":
                best_sim_idx = sim.argmin(0)
            else:
                best_sim_idx = sim.argmax(0)
            _, not_select_idx = torch.topk(sim, sim.shape[0]-1, largest = False)
            doc_to_select.append(best_sim_idx)
        else:
            not_select_idx = np.delete(idx, doc_to_select)
            post_dist_update = post_dist[not_select_idx,:]
            post_selected = post_dist[doc_to_select,:]
            sim_update = sim[not_select_idx]
            for idx_ in range(post_selected.shape[0]):
                if sim_mesure == "cosine":
                    if idx_ == 0:
                        sim_int = F.cosine_similarity(post_selected[idx_,:], post_dist_update, dim=1)  # [num_topics]
                    else:
                        sim_int += F.cosine_similarity(post_selected[idx_,:], post_dist_update, dim=1)  # [num_topics]
                else:
                    if idx_ == 0:
                        sim_int = F.kl_div((post_selected[idx_,:]).log(), post_dist_update, None, None, reduction='none').mean(dim=1)  # [num_topics]
                    else:
                        sim_int += F.kl_div((post_selected[idx_,:]).log(), post_dist_update, None, None, reduction='none').mean(dim=1)  # [num_topics]
                        
            if sim_mesure == "cosine":
                sim_tot = lambda_ * sim_update + (1-lambda_) * sim_int
                best_sim_idx = sim_tot.argmin(0)
            else:
                sim_tot = lambda_ * sim_update - (1-lambda_) * sim_int
                best_sim_idx = sim_tot.argmax(0)
            best_sim_idx = not_select_idx[best_sim_idx]
            doc_to_select.append(torch.tensor(best_sim_idx))
    
    doc_to_select = torch.stack(doc_to_select,dim=0)
    
    return doc_to_select