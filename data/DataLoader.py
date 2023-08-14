#!/usr/bin/env python
# coding: utf-8
from collections import Counter
import spacy
import os
import pandas as pd
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.data_utils import *
from data.Vocab import Vocab
from data.preprocess import TextProcessing

from tqdm import tqdm

PAD_TOKEN = '<pad>'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNK_TOKEN = '<unk>'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '<sos>'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '<eos>'  # This has a vocab id, which is used at the end of untruncated target sequences


class AmazonDataset(Dataset):
    def __init__(self, datapath, max_num_reviews=8, max_len_rev=100, shuffle_revs=False, n_prod_shuf=3, n_fus=0, recons=True, vocab=None, raw_data=False, ban_list = ['DET', 'PUNCT', 'AUX']):
        '''
        Amazon Dataset csv processor for input texts and references to obtain proper batches
        Args:
            datapath: path to the data file
            max_num_reviews: maximum number of reviews in each batch
            max_len_rev: Maximum size of selected reviews - prevent bias effect from singular reviews  -------- PAS L'AIR D'ÊTRE UTILISEE
            shuffle_revs: specify if the reviews should be grouped in a batch independently of the product of not
                n_prod_shuf: If shuffle_revs True, determine from how many different product we sample the reviews in a batch
            n_fus: If shuffle_revs False - determine from how many product the batch should be constituted
            recons: specify if the reference summaries should be loaded (for recontruction loss)
            vocab: vocabulary
            raw_data: Does the input csv file have lready been preprocess - functions in Preprocess_csv folder
        Output:
            reviews_list: List of all reviews in the dataset
            batch_idx_list: List of batch idex - matching the reviews_list
            dict_references: Dictionary containing the dictionary of references for each batch index
            vocab_list: List of all vocaulary for the input reviews
        '''
        super().__init__()
        
        self.nlp = spacy.load("en_core_web_sm")
        self.vocab = vocab
        self.max_len_rev = max_len_rev
        self.ban_list = ban_list
        # Load csv dataset, output text, and batch_ids
        self.src_txt, self.batch_indexes, self.references_dict , self.src_vocab = self.load_dataset(datapath, max_num_reviews, shuffle_revs, load_refs=recons, n_prod_shuf=n_prod_shuf, n_fus=n_fus)
        
        # Preprocess source texts if not preprocessed before from Preprocess_csv folder
        if raw_data:
            processing = TextProcessing()
            print("Preprocessing dataset...")
            for i, sentence in  enumerate(tqdm(self.src_txt)):
                self.src_txt[i] = processing.preprocess(sentence)
        
    def __getitem__(self, index):
        """Forming batch items"""
        #Index is the list indicating the batch size as in load_dataset | for batch_size ==8, then batch_indexes==[0,8,16,...]
        start_idx = self.batch_indexes[index]
        end_idx = self.batch_indexes[index + 1]
        src_batch = []
        src_len_batch = []
        src_bow = []
        batch_size = end_idx - start_idx
        index_others = 0
        
        for index_b in range(start_idx, end_idx):
            txt = self.src_txt[index_b]
            tokens = self.truncate(self.tokenize(txt))
            tokens_bow = self.filtered_tokenize(txt)
            length = len(tokens)
            src_batch.append(tokens)
            src_len_batch.append(length)
            src_bow.append(tokens_bow)
            index_others += 1
            
        return src_batch, src_len_batch, src_bow
    
    def truncate(self, tokens):
        if len(tokens) > self.max_len_rev:
            return tokens[:self.max_len_rev]
        else:
            return tokens
    
    def __len__(self):
        return len(self.batch_indexes) - 1  #Return batch size since it is the number of different product in dataset
    
    def build_vocab(self, vocab_size, min_freq, specials):
        counter = Counter()
        for t in tqdm(self.src_vocab, desc="Build vocabulary"):
        #for t in tqdm(self.src_txt, desc="Build vocabulary"):# - Modif due to shuffe of review per products....
            tokens = self.tokenize(t)
            counter.update(tokens)
        
        
        #sorted counter in from_counter - good for Adaptive Softmax
        self.vocab = Vocab.from_counter(counter=counter, vocab_size=vocab_size, min_freq=min_freq, specials=specials)
        
        return self.vocab
    
    
    def tokenize(self, text):
        return [str(word) for word in self.nlp(str(text))]
    
    def filtered_tokenize(self, text):
        token_text = []
        doc = self.nlp(str(text))
        for token in doc:
            if not token.is_stop and token.pos_ not in self.ban_list:
                token_text.append(str(token.text))
                
        return token_text
            
    def get_references(self):
        return self.references_dict
    
    
    def load_dataset(self, file_path, max_num_reviews=8, shuffle_revs=False, load_refs=False, n_prod_shuf=3, n_fus=0, randomState=42):
        """
        Main function processing and loading data in class
        Args:
            file_path: path to the data file
            max_num_reviews: maximum  number of reviews in each batch | if 0 then no max and all reviews are taken per products
            shuffle_revs: specify if the reviews should be grouped as a batch for each product or not
                n_prod_shuf : number of products shuffled together when shuffle batching
                n_fus:If shuffle_revs False | determine from how many product the batch should be constituted
            load_refs: specify if the reference summaries should be loaded (for recontruction loss)
            randomState: Fixing seed state for data splitting
        Output:
            reviews_list: List of all reviews ordered to be batched
            batch_idx_list: List of start and end index of each batch - if batch_size==8 then list = [0,8,16,...]
            dict_references: Dictionnary with the list of all human generated references | Empty if load_refs=False
            vocab_list: Getting the list of all reviews of all dataset | independently of batches | for vocabulary building
        """
        
        # Read data from file
        df = pd.read_csv(file_path, sep=",")
        col_summ = [col for col in df.columns if col.startswith('summ')]
        prod_ids = list(set(df['prod_id']))
        
        #Getting full vocab from trainning, not depending on what reviews are going to be selected. 
        vocab_list = list(df["review"])
        
        #Storing final results
        reviews_list = []
        batch_idx_list = []
        dict_references = dict() if load_refs else None
        batch_track = 0

        #Shuffling n products together in one batch - topic diversity tests
        if shuffle_revs:
            batch_idx_list.append(0)
    
            #We create shuffled batches while we still have products to deal with
            while len(prod_ids)>0:
                temp_idx = []
                batch_temp = len(batch_idx_list)

                #Managing the last peace of products 
                if len(prod_ids) < n_prod_shuf:
                    n_prod_shuf = len(prod_ids)

                #Selecting n random products
                for i in range(n_prod_shuf):
                    idx = random.randint(0, len(prod_ids)-1)
                    r = prod_ids.pop(idx)
                    temp_idx.append(r)
                    
                #Shuffling the reviews of the n products identified
                df_temp = df[df['prod_id'].isin(temp_idx)]
                if max_num_reviews == 0:
                    df_shuffled = df_temp.sample(frac=1, random_state=randomState)
                else:
                    if n_prod_shuf * max_num_reviews <= len(df_temp):
                        df_shuffled = df_temp.sample(n=n_prod_shuf*max_num_reviews, random_state=randomState)
                    else:
                        df_shuffled = df_temp.sample(frac=1, random_state=randomState)
                review_shuffled = list(df_shuffled["review"])
                reviews_list += review_shuffled   
                
                #Splitting the list of shuffled reviews into n sub lists / batches
                batches = [review_shuffled[i*len(review_shuffled) // n_prod_shuf: (i+1)*len(review_shuffled) // n_prod_shuf] for i in range(n_prod_shuf)]
                for batch in batches:
                    batch_track += len(batch)
                    batch_idx_list.append(batch_track)
                
                #For evaluation - we load the references to estimate ROUGE scores later on
                if load_refs:
                    refs_sum = []
                    for col in col_summ:
                        #Getting the 3 summaries of the first column
                        summs = list(set(df_shuffled[col]))
                        refs_sum.append(summs)

                    #Sampling 3 summaries n times for the n random products included potentially in the batch
                    for i in range(n_prod_shuf):
                        idx = random.randint(0, len(refs_sum)-1)
                        dict_references[batch_temp+i] = refs_sum[idx]
        ################# Probably could be improved to fus n_fus case = 0 and gt 0
        #One batch per product
        else:
            if n_fus > 1:
                #Sampling from n_fus product -- for all products in a range of n_fus we take in the dataframe the n_fus product ids
                for i in tqdm(range(0, len(prod_ids), n_fus), desc="Loading data"):
                    ids = prod_ids[i:i+n_fus]
                    temp = []
                    for id in ids:
                        df_id = df[df['prod_id'] == id]
                        temp.append(df_id)
                    df_temp = pd.concat(temp)
                    result = df_temp.sample(frac=1, random_state=randomState)
                    reviews = list(result["review"])
                    reviews_list += reviews
                    batch_track += len(reviews)
                    batch_idx_list.append(batch_track)
                    batch_temp = len(batch_idx_list) - 1
                    refs_sum = []
                    for col in col_summ:
                        #Getting the 3 summaries of the first column
                        summs = list(set(result[col]))
                        for i, summ in enumerate(summs):
                            if summs != '' and pd.notna(summ):
                                refs_sum.append(summ) 
                    dict_references[batch_temp] = refs_sum
            else: 
                for i, prod_name in tqdm(enumerate(prod_ids), desc="Loading data"):
                    df_temp = df[df['prod_id'] == prod_name]
                    if max_num_reviews == 0 :
                        result = df_temp.sample(frac=1, random_state=randomState)
                    else:
                        if max_num_reviews <= len(df_temp):
                            #If df_temp is not exactly divided by max_num_reviews, we randomly remove reviews to make it perfectly divided
                            n_df = int(len(df_temp) / max_num_reviews)
                            remove_n = len(df_temp) - (n_df * max_num_reviews)
                            drop_indices = np.random.choice(df_temp.index, remove_n, replace=False)
                            # df_temp_dropped is a multiple of max_num_reviews
                            df_temp_dropped = df_temp.drop(drop_indices)
                            #We divide and randomly sample n_df dataframe of max_num_reviews size to create batches
                            df_shuffled = df_temp_dropped.sample(frac=1, random_state=randomState)
                            result = np.array_split(df_shuffled, n_df)
                        else: 
                            result = df_temp.sample(frac=1, random_state=randomState)
                    
                    #Case resulting from the array_split line 249 -- otherwise it's a dataframe form l.251 
                    if type(result) == list:
                        for df_result in result:
                            assert len(df_result) == max_num_reviews
                            reviews = list(df_result["review"])
                            reviews_list += reviews
                            batch_track += len(reviews)
                            batch_idx_list.append(batch_track)
                            if load_refs:
                                batch_temp = len(batch_idx_list) - 1
                                refs_sum = []
                                for col in col_summ:
                                    #Getting the 3 summaries of the first column
                                    summs = list(set(df_result[col]))
                                    if summs[0] != '' and pd.notna(summs):
                                        refs_sum.extend(summs)
                                dict_references[batch_temp] = refs_sum
                    else: 
                        reviews = list(result["review"])
                        reviews_list += reviews
                        batch_track += len(reviews)
                        batch_idx_list.append(batch_track)
                        
                    #HEre we load the summaries of all products included in the shuffle
                    if load_refs:
                        batch_temp = len(batch_idx_list) - 1
                        refs_sum = []
                        for col in col_summ:
                            summs = list(set(df_shuffled[col]))
                            if summs[0] != '' and pd.notna(summs):
                                refs_sum.extend(summs)
                        dict_references[batch_temp] = refs_sum

        assert sum([int(idx <= len(reviews_list)) for idx in batch_idx_list]) == len(batch_idx_list)
        
        return reviews_list, batch_idx_list, dict_references, vocab_list


class Batch:
    """
    Called by Dataloader and create/format elements in batch
    Args : iterable dataset and vocabulary provided by build_dataset function
    """
    def __init__(self, data, vocab, device, vocab_train_size):
        #src, src_len = list(zip(*data)) #receive a list of n text for each 1 element of batch
        src, src_len, src_bow = data[0][0], data[0][1], data[0][2]# , other_src_idxs
        
        self.vocab = vocab
        self.vocab_train_size = vocab_train_size
        self.pad_id = self.vocab.pad()
        self.unk_id = self.vocab.unk() 
        
        # Encoder info
        self.src, self.src_mask, self.src_len, self.group_src = None, None, None, None#, self.idx_non_src, None
        # Additional info for pointer-generator network
        self.mask_others, self.src_others = None, None
        self.src_BoW = None
        
        #Target is modified to src to replace OOVs by random samples OOVs from the same batch in order to allow copy mechanism to learn
        self.tgt_ext = None
        
        # Build batch inputs
        self.init_encoder_seq(src, src_len, src_bow)#, other_src_idxs)

        # Save original strings
        self.src_text = src
        self.to(device)
        
    def init_encoder_seq(self, src, src_len, src_bow):#, other_src_idxs):
        """
        Take source texts and transform into tensors of ids corresponding to the vocabulary
        Transform list of src_len and group_idx into tensors
        """
        #################################################
        # Individual review information 
        #################################################
        #Create list list of token ids corresonding in position in vocabulary
        #Filtering term for learning BoW
        src_BoW_ids = []
        for s in src_bow:
            temp_bow = []
            temp_bow.extend(self.vocab.tokens2ids(s))
            src_BoW_ids.append(temp_bow)
        
        src_ids = []
        for s in src:
            temp = [self.vocab.start()]
            temp.extend(self.vocab.tokens2ids(s))
            temp += [self.vocab.stop()]
            src_ids.append(temp)
            
        #Taking into account the adding of the token in src size
        src_len = [x + 2 for x in src_len]
        self.src_len = torch.LongTensor(src_len)
        
        #Function transforming into tensor, padding it to max_len of batch
        self.src = collate_tokens(values=src_ids, pad_idx=self.pad_id)                                 # [batch_size, seq_len]
        self.src_BoW = collate_tokens(values=src_ids, pad_idx=self.pad_id)                             # [batch_size, seq_len - stopwords and PUNCT etc.] 
        #Creating masks for attention
        self.src_mask = src_mask(self.src, self.pad_id)                                                # [batch_size, seq_len]
        
        # Save additional info for pointer-generator - Determine max number of source text OOVs in this batch
        oovs = [self.vocab.source2ids_ext(s, self.vocab_train_size) for s in src]
        tgt_ext_ids = creat_tgt_ext(src_ids, oovs, self.unk_id)
        self.tgt_ext = collate_tokens(values=tgt_ext_ids, pad_idx=self.pad_id)
        
        #################################################
        # Group review information 
        #################################################
        #Generating general input for all group reviews - Used for attention in calculation of C states
        self.group_src = flat_group(self.src)
        self.group_mask = flat_group(self.src_mask)
        
        #Leave-one-out masking
        self.mask_others = flat_group(self.src_mask, leave_one_out=True)           # Used for direct information access when decoding
        self.src_others = flat_group(self.src, leave_one_out=True)
        
    def __len__(self):
        return self.enc_input.size(0)

    def __str__(self):
        
        """
        Create callable object of the batch - batch.enc_len to get length
        """
        
        batch_info = {
            'src_text': self.src_txt,
            'src': self.src,                                              # [batch_size, seq_len]
            'tgt_ext': self.tgt_ext,                                      # [batch_size, seq_len]
            'src_mask' : self.src_mask,                                   # [batch_size, seq_len]
            'src_len': self.src_len,                                      # [batch_size]
            'group_src' : self.group_src,                                 # [batch_size, total_elements]
            'group_mask' : self.group_mask,                               # [batch_size, total_elements]
            'src_others' : self.src_others,                               # [batch_size, total_elements]
            'mask_others' : self.mask_others,                             # [batch_size, total_elements]
            'src_BoW' : self.src_BoW
        }
        
        return str(batch_info)

    def to(self, device):
        """
        Store object on proper device - cuda or not cuda?
        """
        
        self.src = self.src.to(device)
        self.tgt_ext = self.tgt_ext.to(device)
        self.src_len = self.src_len.to(device)
        self.src_mask = self.src_mask.to(device)
        self.group_src = self.group_src.to(device)
        self.group_mask = self.group_mask.to(device)
        self.src_others = self.src_others.to(device)
        self.mask_others = self.mask_others.to(device)
        self.src_BoW = self.src_BoW.to(device)
        
##################################################################################        
# ========================== END CLASS DEFINITION ================================
def load_create_extended(file_path, vocab_size=2000, vocab_min_freq=1, vocab_ext_path='path', vocab=None, max_num_reviews=0, max_len_rev=100, shuffle_revs=False, n_prod_shuf=3, n_fus=0, recons=False, raw_data=False):
    """
    Creating or Loading external vocabulary | For OOV words and Pointer Generator Network learning
    External vocabulary can be created from the vocabulary from both trainng and validation when vocab will be training vocabulary
    Args:
        file_path: Path toward all reviews csv file to create ext_vocab
        vocab_size: Size of desired vocabulary
        vocab_min_freq: Min frequncy of tokens to be included in vocab | otherwise become <UNK>
        vocab_ext_path: Path toward pickle file with the already generated vocabulary if already existing
        vocab: Initializing vocab to a variable to build it later ----------------------------------------------------- A SUPPRIMER ?? PAS L'AIR D'ÊTRE UTILE
        max_num_reviews | max_len_rev | shuffle_revs | n_prod_shuf | recons: Non impact just for building ext_vocab - let to default values
    Output:
        ext_vocab: Extended vocaulary with bigger vocab than the training one to train <UNK> to predict some tokens in PGN/OOVs
    """
    #Loading extended vocabulary - fix external previously built vocabulary
    if os.path.isfile(vocab_ext_path): 
        vocab_file = open(vocab_ext_path, "rb")
        ext_vocab = pickle.load(vocab_file)
        vocab_file.close()
            
    #Building external vocabulary if not existing
    else:
        #Create dataset processed (tokenized and lowercase) - Field equivalent
        specials = [PAD_TOKEN, UNK_TOKEN, START_DECODING, STOP_DECODING]
        dataset = AmazonDataset(file_path, max_num_reviews, max_len_rev, shuffle_revs, n_prod_shuf, recons, vocab, n_fus, raw_data=raw_data)
        
        #building extended vocabulary
        ext_vocab = dataset.build_vocab(vocab_size=vocab_size, min_freq=vocab_min_freq, specials=specials)
        vocab_file = open(vocab_ext_path, "wb")
        pickle.dump(ext_vocab, vocab_file)
        vocab_file.close()
    
    return ext_vocab

    
def build_dataloader(file_path, vocab_size=2000, vocab_min_freq=1, vocab=None, vocab_train_size=2000, max_num_reviews=8, max_len_rev=100, is_train=True, shuffle_revs=False, n_prod_shuf=3, n_fus=0, shuffle_batch=False, recons=False, device='cpu', batch_size=1, ban_list = ['DET', 'PUNCT', 'AUX'], raw_data=False, vocab_path='path'):
    """
    Generating dataset and Calling Pytorch DataLoader and return the 'iterator' object
    Args: Same
    Output:
        data_loader == Pytorch DataLoader object that will call __getitem__ when called
        vocab_train: Training vocabulary | all individual tokens from training dataset
        references: Reference dictionnary for evaluation
    """
    
    #Create dataset processed (tokenized and lowercase)
    dataset = AmazonDataset(file_path, max_num_reviews, max_len_rev, shuffle_revs, n_prod_shuf, n_fus, recons, vocab, raw_data=raw_data, ban_list = ban_list)
    
    if is_train:  # vocab_train + loading file
        specials = [PAD_TOKEN, UNK_TOKEN, START_DECODING, STOP_DECODING]
        vocab_train = dataset.build_vocab(vocab_size=vocab_size, min_freq=vocab_min_freq, specials=specials)
        vocab_train_size = len(vocab_train)
        vocab_file = open(vocab_path, "wb")
        pickle.dump(vocab_train, vocab_file)
        vocab_file.close()
    else:
        assert vocab is not None
        vocab_train = None
    #Generating dataloader object from the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_batch, collate_fn=lambda data, v=vocab: Batch(data=data, vocab=v, device=device, vocab_train_size=vocab_train_size))
    
    if recons:
        references = dataset.get_references()
    else:
        references = None
    
    return data_loader, vocab_train, references