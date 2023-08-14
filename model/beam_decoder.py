#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import re
from model.Decoder import GRU_Decoder
import torch.distributions.dirichlet as d
import torch.nn.functional as F
from utils.generation import *
from utils.model_utils import filter_oov

class Possible_solutions(object):
    def __init__(self, tokens, log_probs, context_vector, z):
        
        self.tokens = tokens                      #List of tokens
        self.log_probs = log_probs                #List of log_probabilities
        self.context_vector = context_vector
        self.z = z

    def extend(self, token, log_prob, context_vector, z):
        
        return Possible_solutions(tokens=self.tokens + [token],     
                          log_probs=self.log_probs + [log_prob],
                          context_vector=context_vector, z=z)
    
    
    def n_gram_blocking(self, n):
        return self.tokens[-n:]
    
    def mirror_block(self, n):
        return self.tokens[-n-1:]
    
    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class beam_decoder(nn.Module):
    def __init__(self, vocab, embeddings, decoder, copy_linear, lin_inter, max_oov_len, beam_size=5, min_dec_steps=30, num_return_seq=5, num_return_sum=1, n_gram_block=3, mirror_block=2):
        super(beam_decoder, self).__init__()
        
        #General parameters for beam_search and text generation
        self.vocab = vocab
        self.beam_size = beam_size
        self.min_dec_steps = min_dec_steps
        self.num_return_seq = num_return_seq
        self.num_return_sum = num_return_sum
        self.n_gram_block = n_gram_block
        self.mirror_block = mirror_block
        
        #Necessary models for decoding
        self.embedding = embeddings
        self.decoder = decoder
        self.copy_gate = copy_linear
        self.lin_inter = lin_inter
        self.max_oov_len = max_oov_len
        self.mirror_list = ["and", 'or', ',', 'but', '.']
        self.mirror_tokens = [self.vocab.word2id(conj) for conj in self.mirror_list]

    def sort_hyps(self, hyps):
        """Sort hypotheses according to their log probability."""
        return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
    
    def sort_hyps_topic(self, hyps, topic):
        """Sort hypotheses according to their log probability."""
        temp_ = []
        for i, h in enumerate(hyps):
            temp_.append((h, h.avg_log_prob * topic[i], i))
        results = sorted(temp_, key=lambda x: x[1], reverse=True)
        return [i[0] for i in results], [i[2] for i in results]

    def decode_summary(self, src, gen_len, encoder_outputs, z, Bow_outputs, topic_to_gen, src_mask, group_src, group_mask, extra_feat, context_vector, add_extrafeat_pgn):
  
        """
        Beam decoding function
        Args:
            src: [batch_size, seq_len] | Input reviews - just used for having batch_size -- to replace
            gen_len: (int) Lentgh of generated sequences - avg len of batch of input reviews
            encoder_outputs = [batch size, seq_len, group_dim] | Group representations
            z = [batch_size, z_dim]  - already concatenated with topic representation
            group_src = [batch_size, total_elements]
            group_mask = [batch_size, total_elements]
            src_mask = [batch_size, seq_len]
            add_extrafeat_pgn: Wether to account or not for additional extra feature
            extra_feat = [batch_size, z_dim, seq_len]
            context_vector = [batch_size, enc_dim*2] | [batch_size, enc_dim]
            Bow_outputs = [batch_size, vocab_size] | predicted BoW distribution
            topic_to_gen = [vocab_size] | predicted word distribution for the current topic
        Output:
            hyp_results: List of retained summaries - word sequence
            token_list_rencode: List of retained summaries - id sequence | to reuse for re-encoding
        """

        batch_size = src.size(1)
        topic_solutions = []
        best_hyps = []
        
        for idx in range(batch_size):
            #Storing result for specific idx sentence
            sequence_results = []
            #taking all objects dedicated to that sequence
            encoder_outputs_idx = encoder_outputs[idx, :, :].unsqueeze(0)            # [1, seq_len, enc_dim]
            z_idx = z[idx, :].unsqueeze(0)                                           # [1, z_dim]
            src_mask_idx = src_mask[idx, :].unsqueeze(0)                             # [1,seq_len]
            Bow_outputs_idx = Bow_outputs[idx, :].unsqueeze(0)                       # [1, ext_vocab_size]
            group_src_idx = group_src[idx, :].unsqueeze(0)                           # [1, total_elements]
            group_mask_idx = group_mask[idx, :].unsqueeze(0)                         # [1, total_elements]
            extra_feat_idx = extra_feat[idx, :].unsqueeze(0)                         # [1, z_dim]
            context_vector_idx = context_vector[idx, :].unsqueeze(0)                 # [1, enc_dim]
            topic_attention = torch.index_select(topic_to_gen, 0, group_src[0,:])

            #K = number of running hypotheses | Decoding sentence per topics
            hyps = [Possible_solutions(tokens=[self.vocab.start()], log_probs=[0.0], context_vector=context_vector_idx, z=z_idx)]
            for t in range(gen_len):
                num_orig_hyps = len(hyps)  
                input_dec = [hyp.latest_token for hyp in hyps]
                input_dec = torch.tensor(input_dec, dtype=torch.long, device=src.device)            # [K, emb_dim]
                input_embs = self.embedding(input_dec)                                              # [K, emb_dim]
                z_hyp = torch.cat([hyp.z for hyp in hyps], dim=0)                                   # [K, z_dim]
                encoder_outputs_hyp = torch.cat([encoder_outputs_idx for _ in hyps], dim=0)         # [seq_len, K, enc_dim]
                src_mask_hyp = torch.cat([src_mask_idx for _ in hyps], dim=0)                       # [K, seq_len]
                mask_hyp_t = src_mask_hyp[:,t].unsqueeze(-1)                                        
                group_src_hyp = torch.cat([group_src_idx for _ in hyps], dim=0)                     # [1, total_elements]
                group_mask_hyp = torch.cat([group_mask_idx for _ in hyps], dim=0)                   # [1, total_elements]
                extra_feat_hyp = torch.cat([extra_feat_idx for _ in hyps], dim=0)                   # [1, z_dim]
                context_vector_hyp = torch.cat([hyp.context_vector for hyp in hyps], dim=0)         # [1, enc_dim]
                extra_feat_hyp_t = extra_feat_hyp[:,:,t]                                            # [batch_size, z_dim]
                
                ########## Decoder bloc   #########
                vocab_dist, attn_dist, context_vector_hyp, z_hyp = self.decoder(input_embs, encoder_outputs_hyp, z_hyp, group_mask_hyp, mask_hyp_t, context_vector_hyp, extra_feat_hyp_t, topic_attention)
                
                if add_extrafeat_pgn:
                    input_pgn = torch.cat((input_embs, extra_feat_hyp_t, context_vector_hyp, z_hyp), dim=1)              # [K, emb_dim + z_dim + num_topics]
                else:
                    input_pgn = torch.cat((input_embs, context_vector_hyp, z_hyp), dim=1)                                # [K, emb_dim]

                prob_ptr = torch.sigmoid(self.copy_gate(torch.tanh(self.lin_inter(input_pgn))))
                p_gen = 1 - prob_ptr                                                                                                # [K, seq_len]

                #Step 5 - Compute prob dist'n over extended vocabulary
                vocab_dist = p_gen * vocab_dist                                                                                         # [K, output_dim]
                weighted_attn_dist = prob_ptr * attn_dist
                extra_zeros = torch.zeros((num_orig_hyps, self.max_oov_len), device=vocab_dist.device)                                  # [K, OOV_len]
                extended_vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1)                                                      # [K, output_dim + OOV_len]
                final_dist = extended_vocab_dist.scatter_add_(1, group_src_hyp, weighted_attn_dist)                                     # [K, output_dim + OOV_len]
                topic_LM_dist = final_dist
                topic_LM_dist_ = torch.log(topic_LM_dist)

                ############ beam decoding part
                # Take 3*k in size
                topk_prob_LM, topk_ids = torch.topk(topic_LM_dist_, k=self.beam_size * 3, dim=-1)
                topk_probs = torch.log(final_dist)
                all_hyps = []
                for i in range(num_orig_hyps):
                    h_i = hyps[i]
                    context_vector_i = context_vector_hyp[i].unsqueeze(0)                                                             # [1, dec_dim]
                    z_i = z_hyp[i].unsqueeze(0)                                                                                       # [1, z_dim]
                    for j in range(self.beam_size * 3):
                        if topk_ids[i, j].item() == self.vocab.unk():
                            #If despite everything we still generate a unk token, then we sample a token from topic instead
                            new_idx = torch.multinomial(Bow_outputs_idx, 1)
                            new_hyp = h_i.extend(token=new_idx.item(), log_prob=topk_probs[i, j].item(), context_vector=context_vector_i, z=z_i)
                        if t == 0:
                            # Checking sentence don't start by mirror token
                            if topk_ids[i, j].item() in self.mirror_tokens: 
                                continue
                            else:
                                new_hyp = h_i.extend(token=topk_ids[i, j].item(), log_prob=topk_probs[i, j].item(), context_vector=context_vector_i, z=z_i)   
                        else: 
                            #Preventing token repetition
                            if topk_ids[i, j].item() in h_i.n_gram_blocking(self.n_gram_block):
                                continue
                            # Preventing repetition of two consecutive mirror tokens (and .)
                            elif topk_ids[i, j].item() in self.mirror_tokens and h_i.tokens[-1] in self.mirror_tokens:
                                continue
                            # Preventing mirrors [a, b, AND, a, b]
                            elif t >= 2:
                                #[True, False], [False, True], [False, False]
                                presence_mirror = sum(torch.tensor(h_i.mirror_block(self.mirror_block))[-self.mirror_block:]==i for i in self.mirror_tokens).bool() 
                                idx_prev_tokens = presence_mirror.nonzero(as_tuple=True)[0]
                                # Since when [True, False] or [False, True] we alwas check if topk_ids[i, j].item() == h_i.n_gram_blocking[-3] we don't need to different conditions
                                if idx_prev_tokens.shape[0] == 1 and topk_ids[i, j].item() == h_i.mirror_block(self.mirror_block)[-3]:
                                    continue
                                else:
                                    new_hyp = h_i.extend(token=topk_ids[i, j].item(), log_prob=topk_probs[i, j].item(), context_vector=context_vector_i, z=z_i)
                            else:
                                new_hyp = h_i.extend(token=topk_ids[i, j].item(), log_prob=topk_probs[i, j].item(), context_vector=context_vector_i, z=z_i)

                        all_hyps.append(new_hyp)
                #Selecting best terms to continue sequences from LM only | Checking if conditions for beam_search generated sequences have been fulfilled
                hyps = []
                for hyp in self.sort_hyps(all_hyps):
                    if hyp.latest_token == self.vocab.stop():
                        if t >= self.min_dec_steps:
                            sequence_results.append(hyp)
                    else:
                        hyps.append(hyp)
                    if len(hyps) == self.beam_size or len(sequence_results) == self.beam_size:
                        break
                if len(sequence_results) == self.beam_size:
                    break

            # Reached max decode steps but not enough results --> len(sequence_results)
            if len(sequence_results) < self.num_return_seq:
                sequence_results = sequence_results + hyps[:self.num_return_seq - len(sequence_results)]
            
            #Preserving the beam_size final solutions
            best_hyps.extend(self.sort_hyps(sequence_results)[:self.num_return_sum])
            
        #We return the best sequence out of the batch_size * beam_size potential solutions - we consider sequence with best LM+TM probability
        sorted_topic_sent, sorted_index_list = self.sort_hyps_topic(best_hyps, topic_to_gen)
        best_topic_sent, index_list = sorted_topic_sent[:self.num_return_sum], sorted_index_list[:self.num_return_sum]
        
        #Transforming sequence into words
        hyp_words = [self.vocab.outputids2words(best_hyp.tokens, []) for best_hyp in best_topic_sent]
        token_list_rencode = [best_hyp.tokens for best_hyp in best_topic_sent]
        hyp_results = [postprocess(words, skip_special_tokens=True, clean_up_tokenization_spaces=True) for words in hyp_words]

        return hyp_results, token_list_rencode