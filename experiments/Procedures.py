#!/usr/bin/env python
# coding: utf-8
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import time
import json

#Getting metrics on tensorBoard
from torch.utils.tensorboard import SummaryWriter

#Personalized libraries
from utils.model_utils import SeqToBow
from utils.Errors import loss_estimation
from utils.general_utils import frange_cycle_linear, constant_annealing

class Procedure():
    
    def __init__(self, model, ext_vocab, vocab, writer, n_epoch_cycle, optimizer, clip=1.0, learning_rate = 1e-4, weight_decay=0.005, accumulation_steps=1, write_rec_loss=True, write_kl_loss=True, write_BoW_loss=True, write_total_loss=True, z_weight_max=1.0, c_weight_max=0.6, t_weight_max=1.0, topic_encod='GROUP', encode_type='BoW', momentum=0.99, teacher_forcing_ratio=0.9):
        
        #Model parameters
        self.model = model
        self.vocab = vocab
        self.ext_vocab = ext_vocab
        self.momentum = momentum
        #Learning parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip = clip
        self.optimizer = optimizer(params=self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, betas=(self.momentum, 0.99))
        self.accumulation_steps = accumulation_steps
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        #Folowing results through tensorboard
        self.writer = writer
        self.write_rec_loss = write_rec_loss
        self.write_kl_loss = write_kl_loss
        self.write_BoW_loss = write_BoW_loss
        self.write_total_loss = write_total_loss
        
        #Loss parameters
        self.z_weight_max = z_weight_max
        self.c_weight_max = c_weight_max
        self.t_weight_max = t_weight_max
        self.topic_encod=topic_encod
        self.n_epoch_cycle = n_epoch_cycle
        
        #Rouge evaluation parameters
        self.encoder_bow = SeqToBow(ext_vocab, topic_encod=topic_encod, encode_type=encode_type)
        
    def train(self, iterator, iterator_size, epoch, cycle_iter, n_cycles_ct=1, start_lin_t= 0, duration_lin_t = 40):
        
        self.model.train()
        epoch_loss = 0.
        epoch_rec = 0.
        epoch_kl = 0.
        epoch_bow = 0.
        epoch_z = 0.
        epoch_c = 0.
        epoch_t = 0.
        
        #Controling backpropagation acumulation steps to emulate a fix number of element per batch
        bs_cumul = 0 
        
        # Initializing annealing cycles - Defining annealing cycles - gradually increase during 2 epochs before setting back to 0
        cycles = frange_cycle_linear(iterator_size * self.n_epoch_cycle, n_cycle=n_cycles_ct)
        cycle_iter += iterator_size
        #Re-initialize cycle at each n_epoch_cycle
        if epoch % self.n_epoch_cycle == 0:
            cycle_iter = 0
        
        iterator_ = iter(iterator)
        for i in tqdm(range(iterator_size)):
            batch = iterator_.next()
            src  = batch.src                                                          # [batch_size, seq_len]
            src_mask = batch.src_mask                                                 # [batch_size, seq_len]
            src_len = batch.src_len                                                   # [batch_size]
            tgt_ext = batch.tgt_ext                                                   # [batch_size, seq_len]
            group_src = batch.group_src                                               # [batch_size, total_elements = batch_size * src_len]
            group_mask = batch.group_mask                                             # [batch_size, total_elements]
            src_others = batch.src_others                                             # [batch_size, total_elements]
            mask_others = batch.mask_others                                           # [batch_size, total_elements]  - leave one out masking for attention
            batch_size = src.size(0)
            
            #Generating the BoW representations from src
            src_BoW = batch.src_BoW
            BoW_src = self.encoder_bow(src_BoW.permute(1,0), self.vocab.pad())
            
            final_dists, avg_c_kl, avg_z_kl, avg_t_kl, Bow_outputs = self.model(src, src_len, src_mask, group_src, src_others, group_mask, mask_others, BoW_src, teacher_forcing_ratio=self.teacher_forcing_ratio)
                        
            ####### Loss #######
            #Computing annealing ration for z and c (cycle annealing) and t (constant annealing for first epochs)
            cycle_value = cycles[i+cycle_iter]
            t_value = constant_annealing(iterator_size, epoch, i, self.t_weight_max,  start= start_lin_t, duration= duration_lin_t)
            
            #Reshaping output distribution for error computation
            vocab_dim = final_dists.shape[-1]
            seq_len = final_dists.shape[0]
            final_dists = final_dists[1:].reshape((seq_len-1)*batch_size, vocab_dim)                                         # [(seq_len -1) * batch_size, ext_vocab]
            tgt_ext = tgt_ext.permute(1,0)[1:].reshape((seq_len-1)*batch_size)                                               # [(seq_len -1) * batch_size]
            
            #Loss function
            rec_loss, kl_loss, BoW_loss, kld_c, kld_z, kld_t = loss_estimation(final_dists, avg_c_kl, avg_z_kl, avg_t_kl, tgt_ext, self.ext_vocab, BoW_src, Bow_outputs, cycle=cycle_value, annealing_t=t_value, z_weight_max=self.z_weight_max, c_weight_max=self.c_weight_max, t_weight_max=self.t_weight_max)
            
            loss = BoW_loss + kl_loss + rec_loss
            loss.register_hook(lambda grad: grad/batch_size)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            
            bs_cumul += batch_size
            if bs_cumul >= self.accumulation_steps:
                self.optimizer.step()
                self.optimizer.zero_grad()
                bs_cumul = 0
            
            epoch_loss += loss.item()
            epoch_rec += rec_loss.item()
            epoch_kl += kl_loss.item()
            epoch_bow += BoW_loss.item()
            epoch_z += kld_z.item()
            epoch_c += kld_c.item()
            epoch_t += kld_t.item()

        ####### Tensorboard statistics recording
        if self.writer != None:
            if self.write_rec_loss: 
                self.writer.add_scalar("Train/rec_loss", epoch_rec / len(iterator), global_step=epoch)
            if self.write_kl_loss: 
                self.writer.add_scalar("Train/kl_loss", epoch_kl / len(iterator), global_step=epoch)
            if self.write_BoW_loss:
                self.writer.add_scalar("Train/BoW_loss", epoch_bow / len(iterator), global_step=epoch)
            if self.write_total_loss:
                self.writer.add_scalar("Train/total_loss", epoch_loss / len(iterator), global_step=epoch)
        
        return epoch_loss / iterator_size

    def evaluation(self, iterator, iterator_size, epoch, cycle_iter, references=None, save_name=None, start_lin_t= 0, duration_lin_t = 40, gen_len_method="NSumm", topic_gen="select_main", gen_word=[], n_cycles_ct=1, num_t_gen=3, t_mask_len=8, rencode=True):

        self.model.eval()
        
        epoch_loss = 0.0
        epoch_rec = 0.0
        epoch_kl = 0.0 
        epoch_bow = 0.0
        tot_results = {}

        # Initializing annealing cycles
        cycles = frange_cycle_linear(iterator_size * self.n_epoch_cycle, n_cycle=n_cycles_ct)
        cycle_iter += iterator_size
        if epoch % self.n_epoch_cycle == 0:
            cycle_iter = 0
        
        with torch.no_grad():
            batch_hidden = []
            iterator_ = iter(iterator)
            for i in tqdm(range(iterator_size)):
                batch = iterator_.next()
                src  = batch.src                               # [batch_size, seq_len]
                src_mask = batch.src_mask                      # [batch_size, seq_len]
                src_len = batch.src_len                        # [batch_size]
                tgt_ext = batch.tgt_ext                        # [batch_size, seq_len]
                group_src = batch.group_src                    # [batch_size, total_elements = batch_size * src_len]
                group_mask = batch.group_mask                  # [batch_size, total_elements]
                src_others = batch.src_others                  # [batch_size, total_elements]
                mask_others = batch.mask_others                # [batch_size, total_elements]  - leave one out masking for attention
                src_BoW = batch.src_BoW
                BoW_src = self.encoder_bow(src_BoW.permute(1,0), self.vocab.pad())
                batch_size = src.size(0)
                final_dists, avg_c_kl, avg_z_kl, avg_t_kl, Bow_outputs = self.model(src, src_len, src_mask, group_src, src_others, group_mask, mask_others, BoW_src, teacher_forcing_ratio=self.teacher_forcing_ratio)
                
                
                ####### Loss #######
                cycle_value = cycles[i+cycle_iter]
                t_value = constant_annealing(iterator_size, epoch, i, self.t_weight_max,  start= start_lin_t, duration= duration_lin_t)
                
                vocab_dim = final_dists.shape[-1]
                seq_len = final_dists.shape[0]
                final_dists = final_dists[1:].reshape((seq_len-1)*batch_size, vocab_dim)                      # [(seq_len -1) * batch_size, ext_vocab]
                tgt_ext = tgt_ext.permute(1,0)[1:].reshape((seq_len-1)*batch_size)                            # [(seq_len -1) * batch_size]
                
                rec_loss, kl_loss, BoW_loss, kld_c, kld_z, kld_t = loss_estimation(final_dists, avg_c_kl, avg_z_kl, avg_t_kl, tgt_ext, self.ext_vocab, BoW_src, Bow_outputs, cycle=cycle_value, annealing_t=t_value, z_weight_max=self.z_weight_max, c_weight_max=self.c_weight_max, t_weight_max=self.t_weight_max)
                
                loss = rec_loss + BoW_loss + kl_loss
                
                epoch_loss += loss.item()
                epoch_rec += rec_loss.item()
                epoch_kl += kl_loss.item()
                epoch_bow += BoW_loss.item()
                
                #Saving results for later evaluation
                if save_name is not None:
                    #Reconstructing reviews and summaries for ROUGE evaluation
                    summ_sentences, hidden_reps = self.model.inference(src, src_len, src_mask, group_src, group_mask, BoW_src, gen_len_method=gen_len_method, topic_gen=topic_gen, gen_word=gen_word, num_t_gen=num_t_gen, t_mask_len=t_mask_len, rencode=rencode)

                    #Saving results for each batch with the corresponding group of the reviews
                    reviews = batch.src_text
                    tot_results[i+1] = {'summaries' : {}, 'human_refs' : {}, 'orig_reviews' : {}}
                    #appending generated sentences
                    for j in range(len(summ_sentences)):
                        tot_results[i+1]['summaries']['gen_{}'.format(j)] = summ_sentences[j][0]
                    #appending references
                    if references != None:
                        batch_ref = references[i+1]
                        for j in range(len(batch_ref)):
                            tot_results[i+1]['human_refs']['ref_{}'.format(j)] = batch_ref[j]
                    for j in range(len(reviews)):
                        tot_results[i+1]['orig_reviews']['src_{}'.format(j)] = ' '.join(reviews[j])
                    
                    batch_hidden.append(hidden_reps)
                
            ######Tensorboard statistics recording
            if self.writer != None:
                if self.write_rec_loss:
                    self.writer.add_scalar("Valid/rec_loss", epoch_rec / len(iterator), global_step=epoch)
                if self.write_kl_loss:
                    self.writer.add_scalar("Valid/kl_loss", epoch_kl / len(iterator), global_step=epoch)
                if self.write_BoW_loss:
                    self.writer.add_scalar("Valid/BoW_loss", epoch_bow / len(iterator), global_step=epoch)
                if self.write_total_loss:
                    self.writer.add_scalar("Valid/total_loss", epoch_loss / len(iterator), global_step=epoch)
            
                    
            if save_name is not None:
                results = json.dumps(tot_results, indent = 2)
                jsonFile = open(f"experiments/results/results_{save_name}_valid.json", "w")
                jsonFile.write(results)
                jsonFile.close()                 
                # Save hidden representations z to file for diversity analysis
                torch.save(batch_hidden, f'experiments/results/hidden_{save_name}_valid.pt')
                print(f'Files {save_name} saved')

        return epoch_loss / iterator_size

    
    def generation(self, iterator, iterator_size, epoch, cycle_iter, references=None, gen_len_method="NSumm", topic_gen="select_main", gen_word=[], save_name=None, num_t_gen=3, t_mask_len=8, rencode=True):
        
        self.model.eval()
        #Storing final results
        tot_results = {}
        assert save_name is not None, "You need to provide a valid File name (str)"
            
        with torch.no_grad():
            batch_hidden = []
            iterator_ = iter(iterator)
            for i in tqdm(range(iterator_size)):
                batch = iterator_.next()
                src  = batch.src
                src_mask = batch.src_mask
                src_len = batch.src_len                        
                group_src = batch.group_src
                group_mask = batch.group_mask
                src_others = batch.src_others
                mask_others = batch.mask_others
                src_BoW = batch.src_BoW
                BoW_src = self.encoder_bow(src_BoW.permute(1,0), self.vocab.pad())
                
                #Reconstructing reviews and summaries for evaluation
                summ_sentences, hidden_reps = self.model.inference(src, src_len, src_mask, group_src, group_mask, BoW_src, gen_len_method=gen_len_method, topic_gen=topic_gen, gen_word=gen_word, num_t_gen=num_t_gen, t_mask_len=t_mask_len, rencode=rencode)

                #Saving results for each batch with the corresponding group of the reviews
                reviews = batch.src_text
                tot_results[i+1] = {'summaries' : {}, 'human_refs' : {}, 'orig_reviews' : {}}
                #appending generated sentences
                for j in range(len(summ_sentences)):
                    tot_results[i+1]['summaries']['gen_{}'.format(j)] = summ_sentences[j][0]
                #appending references
                if references != None:
                    batch_ref = references[i+1]
                    for j in range(len(batch_ref)):
                        tot_results[i+1]['human_refs']['ref_{}'.format(j)] = batch_ref[j]
                for j in range(len(reviews)):
                    tot_results[i+1]['orig_reviews']['src_{}'.format(j)] = ' '.join(reviews[j])
                
                batch_hidden.append(hidden_reps)
            
            #Saving the final results for the generation
            results = json.dumps(tot_results, indent = 2)
            jsonFile = open(f"experiments/results/results_{save_name}_gen.json", "w")
            jsonFile.write(results)
            jsonFile.close()
            # Save hidden representations z to file for diversity analysis
            torch.save(batch_hidden, f'experiments/results/hidden_{save_name}_gen.pt')
            print(f'Files {save_name} saved')