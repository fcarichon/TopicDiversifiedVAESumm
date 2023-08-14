#!/usr/bin/env python
# coding: utf-8

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.dirichlet as d

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#Specific function needed for copycat
from model.Encoder import GRU_Encoder, BoWEncoder
from model.Decoder import GRU_Decoder, BowDecoder
from model.Attentions import Group_Attention, Attention
from model.Latents_param import mu_sigma_qc, mu_sigma_qz, mu_sigma_pz, mu_sigma_qt
from model.KLD_terms import kld_gauss, kld_normal, kld_dirichlet
from model.beam_decoder import beam_decoder
from utils.model_utils import *

class VAEMultiSumm(nn.Module):

    def __init__(self, ext_vocab, vocab, emb_dim, enc_dim, dec_dim, attn_dim, c_dim, z_dim, num_topics, dropout, device, topic_prior_mean, topic_prior_variance, encod_bidir=True, use_pretrained=True, gen_cond='X', BoW_train=True, Glove_name='6B', Glove_dim=100, beam_size=5, min_dec_steps=30, num_return_seq=5, num_return_sum=1, n_gram_block=3, add_extrafeat=False, add_extrafeat_pgn=False, context_vector_input=True, use_topic_attention=False): 
        
        super(VAEMultiSumm, self).__init__()
        
        """
        Main Model "VAEMultiSumm" to generate topic biased summaries
        Model Parameters : See config.py in configs to have detailed info on parameters and their potential values
        """
        ############################################################
        ############ Initializing General parameters
        self.ext_vocab = ext_vocab
        self.vocab = vocab
        self.max_oov_len = len(ext_vocab) - len(vocab)
        self.src_pad_idx = vocab.pad()
        self.use_pretrained = use_pretrained
        self.input_dim = len(vocab)
        self.output_dim = len(vocab)
        self.device = device
        self.num_return_sum = num_return_sum
        self.add_extrafeat_pgn = add_extrafeat_pgn
        self.add_extrafeat = add_extrafeat
        self.context_vector_input = context_vector_input
        self.use_topic_attention = use_topic_attention

        ############################################################
        ############ Initializing Embedding layers
        self.emb_dim = emb_dim
        if self.use_pretrained:
            emb_vecs = load_gloveembeddings(ext_vocab, Glove_name=Glove_name, Glove_dim=Glove_dim)
            self.embedding = nn.Embedding.from_pretrained(emb_vecs, freeze=False, padding_idx=self.src_pad_idx)
        else:
            self.embedding = nn.Embedding(len(ext_vocab), emb_dim, padding_idx=self.src_pad_idx)
        
        ############################################################
        ############ Initializing Encoders
        self.enc_dim = enc_dim
        self.encod_bidir = encod_bidir
        if encod_bidir:
            group_dim = emb_dim + enc_dim*2
        else:
            group_dim = emb_dim + enc_dim
        self.attn_dim = attn_dim
        self.encoder = GRU_Encoder(self.input_dim, emb_dim, enc_dim, dec_dim, dropout=dropout, bidir=encod_bidir)
        self.gp_attn_layer = Group_Attention(group_dim, attn_dim)
        self.encoder_BoW = BoWEncoder(len(self.ext_vocab), enc_dim, dec_dim, dropout=dropout)
        
        ############################################################
        ############ Initializing C and Z calculation
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.mu_sigma_qc = mu_sigma_qc(group_dim, c_dim)
        self.mu_sigma_qz = mu_sigma_qz(dec_dim, c_dim, z_dim)
        self.mu_sigma_pz = mu_sigma_pz(c_dim, z_dim)
        
        ############################################################
        ############ Initializing Topic t calculation
        self.gen_cond = gen_cond
        self.num_topics = num_topics
        self.topic_to_c = nn.Linear(c_dim + num_topics, c_dim)
        self.topic_to_z = nn.Linear(z_dim + num_topics, z_dim + num_topics)
        self.pt_mu = torch.tensor([topic_prior_mean] * num_topics).to(self.device)   
        self.pt_mu = nn.Parameter(self.pt_mu)
        self.pt_sigma = torch.tensor([topic_prior_variance] * num_topics).to(self.device)
        self.pt_sigma = nn.Parameter(self.pt_sigma)
        self.mu_sigma_qt = mu_sigma_qt(dec_dim, num_topics)
        self.drop_qt = nn.Dropout(dropout)
        #Multinomial prior ford words distributions from topics : 
        self.beta = nn.Parameter(torch.Tensor(num_topics, len(ext_vocab)))                                        # [num_topics, ext_vocab_size]
        nn.init.xavier_uniform_(self.beta)
        self.beta_batchnorm = nn.BatchNorm1d(len(ext_vocab), affine=False)
        
        ############################################################
        ############ Initializaing Decoders and Pointer Generator
        self.dec_dim = dec_dim
        if encod_bidir:
            self.decoder = GRU_Decoder(self.output_dim, emb_vecs, emb_dim, enc_dim*2, z_dim+num_topics, attn_dim, z_dim, dropout, context_vector_input=self.context_vector_input, add_extrafeat=self.add_extrafeat, use_topic_attention=self.use_topic_attention)
            self.w_context = nn.Linear(enc_dim*2, 1, bias=False)
        else:
            self.decoder = GRU_Decoder(self.output_dim, emb_vecs, emb_dim, enc_dim, z_dim+num_topics, attn_dim, z_dim, dropout, context_vector_input=self.context_vector_input, add_extrafeat=self.add_extrafeat, use_topic_attention=self.use_topic_attention)
            self.w_context = nn.Linear(enc_dim, 1, bias=False)
        
        self.w_hidden = nn.Linear(z_dim+num_topics, 1, bias=False)
        self.w_input = nn.Linear(emb_dim + z_dim + num_topics, 1, bias=True)
        self.copy_gate = nn.Linear(enc_dim, 1, bias=False)
        if self.add_extrafeat_pgn:
            self.lin_inter = nn.Linear(emb_dim + z_dim + num_topics + z_dim+num_topics + enc_dim*2, enc_dim)
        else:
            self.lin_inter = nn.Linear(emb_dim + z_dim+num_topics + enc_dim*2, enc_dim)
        self.BoW_train = BoW_train
        
        ############################################################
        ############ Initializing beam decoder for inference
        self.n_gram_block = n_gram_block
        self.beam_decoder = beam_decoder(ext_vocab, self.embedding, self.decoder, self.copy_gate, self.lin_inter, self.max_oov_len, beam_size=5, min_dec_steps=30, num_return_seq=5, num_return_sum=1, n_gram_block=3, mirror_block=2)
    

    def forward(self, src, src_len, src_mask, group_src, src_others, group_mask, mask_others, BoW_src, teacher_forcing_ratio=0.85):
        """
        Training function
        Args:
            src = [batch_size, seq_len] | Input tensors of id per reviews
            src_len = [batch_size] | Length of each review
            src_mask = [batch_size, seq_len]
            group_src = [batch_size, total_elements = batch_size * src_len]
            src_others = [batch_size, total_elements]
            group_mask = [batch_size, total_elements]
            mask_others = [batch_size, total_elements]
            BoW_src = [batch_size, ext_vocab_size]  | Input Bag of Words representations
            teacher_forcing_ratio: Ratio of using true label instead of predicted in training 
        Output:
            final_dists: list of distribution over vocabulary for each token | [gen_len, batch_size, ext_vocab]
            Bow_outputs: Distribution over the vocabulary for presence of terms in reviews | [batch_size, ext_vocab]
            avg_c_kl_term: (int) | Average KL Divergence values for latent variable c
            avg_z_kl_term: (int) | Average KL Divergence values for latent variable z
            avg_t_kl_term: (int) | Average KL Divergence values for latent variable t
            
        """
        src = src.permute(1,0)                                                                                     # [seq_len, batch_size]
        seq_len = src.shape[0]
        src_max_len = torch.tensor(src_len.max()).repeat(src_len.size())                                           # [batch_size]
        batch_size = src.size(1)
        embeddings = self.embedding(src)                                                                           # [seq_len, batch_size, emb_dim]

        ###################################
        # Encoding
        # hidden = [batch_size, dec_dim] / encoder_outputs = [seq_len, batch_size, enc_dim*2]
        encoder_outputs, hidden = self.encoder(embeddings, src_max_len)
        hidden_BoW = self.encoder_BoW(BoW_src)                                                                     # [batch_size, dec_dim]
        
        ###################################
        # C calculations 
        #Step 1 : Get a representation of all the words of the group for each element of the batch
        encoder_outputs = encoder_outputs.permute(1, 0, 2)                                                         # [batch_size, seq_len, enc_dim] or *2 if bidir
        encoder_outputs = encoder_outputs.reshape(batch_size * seq_len, -1)                                        # [tot_elem, enc_dim]
        encoder_outputs = encoder_outputs.unsqueeze(0).repeat(batch_size,1,1)                                      # [batch_size, tot_elem, enc_dim]
        
        #Step 2 : Concat all reviews representations
        group_embeddings = self.embedding(group_src)                                                               # [batch_size, total_els_group, emb_dim]
        group_encoding = torch.cat((encoder_outputs, group_embeddings), dim = 2)                                   # [batch size, seq_len, group_dim]

        #Step 3 : Apply attentions to get group weights
        c_states_scoring = self.gp_attn_layer(group_encoding, group_mask)
        c_states_scoring = c_states_scoring.unsqueeze(2)                                                           # [batch_size, total_els_group, 1]
        group_context = (group_encoding * c_states_scoring).sum(dim=1)                                             # [batch_size, group_dim]

        #Step 4 : We use c_states_scoring et a linear transformation to sample c
        mu_qc, sigma_qc = self.mu_sigma_qc(group_context)                                                          # [batch_size, c_dim]
        qc = mu_qc + torch.randn_like(sigma_qc**0.5) * sigma_qc**0.5                                               # [batch_size, c_dim]
        
        ###################################
        # Topic t calculations - ProdLDA model - https://github.com/estebandito22/PyTorchAVITM/blob/master/pytorchavitm/avitm/decoder_network.py
        #Sampling our posteriors topics
        mu_qt, sigma_qt, log_sigma_qt = self.mu_sigma_qt(hidden_BoW)                                               # [batch_size, num_topics], [batch_size, num_topics]
        qt = F.softmax(mu_qt + torch.randn_like(sigma_qt**0.5) * sigma_qt**0.5, dim=1)                             # [batch_size, num_topics]
        qt = self.drop_qt(qt)
        
        # Bag Of Words Decoder and Error #ProdLDA model as in ATVIM
        if self.BoW_train:                                                                  
            Bow_outputs = F.softmax(self.beta_batchnorm(torch.matmul(qt, self.beta)), dim=1)                       # [batch_size, ext_vocab_size]
        else: 
            Bow_outputs = None
        # If we want to train model decoder directly pondering attention with the topic distribution
        topic_elem = None
        if self.use_topic_attention:
            # Has to be BOW here to have [batch_size, vocab_size for attention weights]
            topic_elem = Bow_outputs
        
        ###################################
        # Z calculations - estimating z from encoder representation and c
        #Determining here if we want to test implementing topic bias into z representation or not
        if self.gen_cond == 'Z' or self.gen_cond == 'BOTH':
            code = torch.cat([qc, qt], dim=1)                                                                      # [batch_size, c_dim + num_topics] 
            code = self.topic_to_c(code)                                                                           # [batch_size, c_dim]
            z_states = torch.cat((hidden, code), dim=-1)                                                           # [batch_size, dec_dim + c_dim]
        else: 
            z_states = torch.cat((hidden, qc), dim=-1)                                                             # [batch_size, dec_dim + c_dim]
        mu_qz, sigma_qz = self.mu_sigma_qz(z_states)                                                               # [batch_size, z_dim]
        qz = mu_qz + torch.randn_like(sigma_qz**0.5) * sigma_qz**0.5                                               # [batch_size, z_dim]
        hidden_qz = qz
        
        ###################################
        # Decoding
        #Transforming qz into  the concatenation of qz and qt for topic bias into generation directly as in paper
        if self.gen_cond == 'X' or self.gen_cond == 'BOTH':
            qz_concat = torch.cat([qz, qt], dim=1)                                                                 # [batch_size, z_dim+num_topics]
            qz_concat = torch.tanh(self.topic_to_z(qz_concat))
            hidden_qz = qz_concat
                
        ###################################
        # Decoding steps
        final_dists = [torch.zeros(batch_size, self.output_dim + self.max_oov_len).to(self.device)]
        if self.encod_bidir:
            context_vector = torch.zeros(batch_size, self.enc_dim*2).to(self.device)                               # [batch_size, enc_dim]
        else:
            context_vector = torch.zeros(batch_size, self.enc_dim).to(self.device)                                 # [batch_size, enc_dim]
        input_dec = src[0,:]                                                                                       # [batch_size]
        extra_feat = qz.unsqueeze(-1).repeat(1, 1, seq_len)                                                        # [batch_size, z_dim, seq_len]
        gen_len = src_len.max()
        
        #Generating steps
        for t in range(1, gen_len):
            input_embs = self.embedding(input_dec)                                                                 # [batch_size, emb_dim]
            mask_t = src_mask[:,t].unsqueeze(-1)                
            extra_feat_t = extra_feat[:,:,t]                                                                       # [batch_size, z_dim]
            
            #Step 1 - decoder
            vocab_dist, attn_dist, context_vector, hidden_qz = self.decoder(input_embs, encoder_outputs, hidden_qz, mask_others, mask_t, context_vector, extra_feat_t, topic_elem)
            
            #Step 2 - Concat feature vectors
            if self.add_extrafeat_pgn:
                input_pgn = torch.cat((input_embs.squeeze(0), extra_feat_t, context_vector, hidden_qz), dim=1)     # [batch_size, emb_dim + z_dim + num_topics]
            else:
                input_pgn = torch.cat((input_embs.squeeze(0), context_vector, hidden_qz), dim=1)                   # [batch_size, emb_dim]
                
            prob_ptr = torch.sigmoid(self.copy_gate(torch.tanh(self.lin_inter(input_pgn))))
            p_gen = 1 - prob_ptr                                                                                   # [batch_size, 1]
            
            #Step 3 - Compute prob dist'n over extended vocabulary
            vocab_dist = p_gen * vocab_dist                                                                        # [batch_size, output_dim]
            weighted_attn_dist = prob_ptr * attn_dist                                                              # [batch_size, total_elem_len]
            
            #Step 4 - manage OOV words with extra_zeros
            extra_zeros = torch.zeros((batch_size, self.max_oov_len),  device=vocab_dist.device)                   # [batch_size, OOV_len]
            extended_vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1)                                     # [batch_size, output_dim + OOV_len]
            final_dist = extended_vocab_dist.scatter_add(dim=-1, index=src_others, src=weighted_attn_dist)
            final_dist = final_dist + Bow_outputs
            
            #Step 5 - append in list for end output and error calculation
            final_dists.append(final_dist)
            
            #Step 6 - Teacher forcing
            top1 = final_dist.argmax(1)                                                                            # [batch_size]
            teacher_force = random.random() < teacher_forcing_ratio
            input_dec = src[t] if teacher_force else top1

        final_dists = torch.stack(final_dists, dim=-1)                                                             # [batch_size, ext_vocab, gen_len]
        final_dists = final_dists.permute(2, 0, 1)                                                                 # [gen_len, batch_size, ext_vocab]
        
        ###################################
        # Error terms
        ### KL term for c
        c_kl_term = kld_normal(mu_qc, sigma_qc, dim=1)                                                             # [batch_size]
        avg_c_kl_term = c_kl_term.mean()
        
        ### KL term for t
        t_kl_term = kld_dirichlet(mu_qt, sigma_qt, log_sigma_qt, self.pt_mu, self.pt_sigma, self.num_topics)
        avg_t_kl_term = t_kl_term.mean()
        
        ### KL term for z
        mu_pz, sigma_pz = self.mu_sigma_pz(c=qc)                                                                   # [batch_size, z_dim]
        z_kl_term = kld_gauss(mu_qz, sigma_qz, mu_pz, sigma_pz)                                                    # [batch_size]
        avg_z_kl_term = z_kl_term.mean()
        
        return final_dists, avg_c_kl_term, avg_z_kl_term , avg_t_kl_term, Bow_outputs
    
    
    def inference(self, src, src_len, src_mask, group_src, group_mask, BoW_src, gen_len_method="NSumm", topic_gen="select_main", gen_word=[], num_t_gen=3, t_mask_len=8, rencode=True):
        
        """
        Inference function
        Args:
            src = [batch_size, seq_len] | Input tensors of id per reviews
            src_len = [batch_size] | Length of each review
            src_mask = [batch_size, seq_len]
            group_src = [batch_size, total_elements = batch_size * src_len]
            group_mask = [batch_size, total_elements]
            mask_others = [batch_size, total_elements]
            BoW_src = [batch_size, ext_vocab_size] | Input Bag of Words representations
            num_t_gen: Number of different biased summaries to generate 
            gen_len_method: SingleSumm = you generate one summary by concatenating num_t_gen summaries of size gen_len/num_t_gen | NSumm = you generate num_t_gen summaries of size n for each topic
            topic_gen: word_bias= You bias generation with input words by user | select_main = You use prior topic distribution to identify the main num_t_gen topic
                IF topic_gen == word_bias:
                    gen_word: List of words to input for the user - Can't be empty
            t_mask_len: What proportion of words have to be masked for biasing group attention -- here we keep 1/8 of owrds then
        Output:
            topic_summs: List of generated summaries
            hiddens: Hidden representations of summaries -- for evaluating diversity of generated texts only -- can be ignore otherwise
        """
        
        src = src.permute(1,0)                                                                                     # [seq_len, batch_size]
        seq_len = src.shape[0]
        embeddings = self.embedding(src)
        src_max_len = torch.tensor(src_len.max()).repeat(src_len.size())                                           # [batch_size]
        batch_size = src.size(1)
        
        #Gen_method to gen_len
        if gen_len_method == "SingleSumm":
            gen_len = int(torch.mean(src_len.float()) / num_t_gen)
        elif gen_len_method == "NSumm":
            gen_len = int(torch.mean(src_len.float()))
        else:
            raise Exception("You need to input a valid generation method (SingleSumm or NSumm) for generating 1 summary of n_subtopics or N summaries by topic")
            
        
        encoder_outputs, hidden = self.encoder(embeddings, src_max_len)
        hidden_BoW = self.encoder_BoW(BoW_src)                                                                     # [batch_size, dec_dim]

        #Initializing & Sampling topics
        mu_t, sigma_t, log_sigma_t = self.mu_sigma_qt(hidden_BoW)                                                  # [batch_size, num_topics], [batch_size, num_topics]
        beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)                                                    # [num_topics, vocab_size]
        t = F.softmax(mu_t + torch.randn_like(sigma_t**0.5) * sigma_t**0.5, dim=1)                                 # [batch_size, num_topics]    
        t = self.drop_qt(t)
        Bow_outputs = F.softmax(self.beta_batchnorm(torch.matmul(t, self.beta)), dim=1)                            # [batch_size, vocab_size]
        beta_post = torch.matmul(t.T, BoW_src)                                                                     # [num_topics, ext_vocab]
        beta_post_soft = F.softmax(beta_post)
        
        if topic_gen == "word_bias":
            assert len(gen_word) > 0, "If you set topic_gen=0, you must provide a list of term to generate summary oriented to those terms."
            select_val = []
            for word in gen_word:
                main_index = self.ext_vocab.word2id(word)
                beta_word = beta_post_soft[:,main_index]
                topic_word = beta_word.argmax()
                select_val.append(topic_word)
            main_topics = torch.tensor(select_val)
        
        elif topic_gen == "select_main":
            prior_t = F.softmax(self.pt_mu + torch.randn_like(self.pt_sigma**0.5) * self.pt_sigma**0.5)            # [1, num_topics]
            main_topics = get_MMR_dist(t, prior_t, sim_mesure="cosine", n=num_t_gen, lambda_=0.2).to(src.device)
        
        else:
            raise Exception("You need to input a valid method (word_bias or select_main) for generating summaries oriented by the topics")

        encoder_outputs = encoder_outputs.permute(1, 0, 2)                                                         # [batch_size, seq_len, enc_dim] or *2 if bidir
        encoder_outputs = encoder_outputs.reshape(batch_size * seq_len, -1)                                        # [tot_elem, enc_dim]
        encoder_outputs = encoder_outputs.unsqueeze(0).repeat(batch_size,1,1)                                      # [batch_size, tot_elem, enc_dim]
        group_embeddings = self.embedding(group_src)                                                               # [batch_size, total_els_group, emb_dim]
            
        
        if self.encod_bidir:
            context_vector = torch.zeros(batch_size, self.enc_dim*2).to(self.device)                               # [batch_size, enc_dim]
        else:
            context_vector = torch.zeros(batch_size, self.enc_dim).to(self.device)
        
        hidden_sums = []
        topic_summs = [] 
        for topic in main_topics:
            topic_to_gen = t[:,topic]                                                                              # [batch_size]
            #Setting all topic to the topic to generate
            t_ = topic_to_gen.unsqueeze(-1).repeat(1, self.num_topics)                                             # [batch_size, num_topics]
            beta_topic = beta_post[topic]                                                                          # [vocab_size]
            #Create tensor of topic values for each word include in group_src
            topic_elem_ = torch.index_select(beta_topic, 0, group_src[0,:])                                        # [total_elem]
            topic_elem = topic_elem_.unsqueeze(0).repeat(batch_size, 1)                                            # [batch_size, total_elem]

            
            #####################################################################################################
            # Modification of group_msaking to account only for topic related documents - we first decide to keep only half the batch
            # Get index of documents with smallest topic related values and masking their input in group_mask
            _, topic_words = torch.topk(topic_elem_, k=int(topic_elem_.shape[0]/t_mask_len), largest=False)
            group_mask_ = torch.empty_like(group_mask).copy_(group_mask)
            group_mask_[:, topic_words] = False
            group_src = group_src * group_mask_
            
            #Estimation of C states
            group_encoding = torch.cat((encoder_outputs, group_embeddings), dim = 2)                               # [batch size, seq_len, group_dim]
            c_states_scoring = self.gp_attn_layer(group_encoding, group_mask_)
            # We weight each word of the total element by it's distribution in the considered topic which act as a first attention by topic
            group_context = (group_encoding * c_states_scoring.unsqueeze(2)).sum(dim=1)
            #We create a mean representation toward the topic by increasing the weight of indiv document representing the most the topic
            mu_qc, sigma_qc = self.mu_sigma_qc(group_context)
            qc = mu_qc

            #Estimation of z states to its prior mean biased by topics
            mu_pz, sigma_pz = self.mu_sigma_pz(c=qc)                                                               # [batch_size, z_dim]
            topic_mu_pz = torch.sum(topic_to_gen.unsqueeze(-1) * mu_pz, dim=0)
            topic_mu_pz = topic_mu_pz.unsqueeze(0).repeat(batch_size,1)
            
            if self.gen_cond == 'X' or self.gen_cond == 'BOTH':
                z_temp = torch.cat([topic_mu_pz, t_], dim=1)
                z = torch.tanh(self.topic_to_z(z_temp))
            
            extra_feat = mu_pz.unsqueeze(-1).repeat(1, 1, seq_len)
            summ_sentences, id_list = self.beam_decoder.decode_summary(src, gen_len, encoder_outputs, z, Bow_outputs, beta_topic, src_mask, group_src, group_mask, extra_feat, context_vector, self.add_extrafeat_pgn)
            
            topic_summs.append(summ_sentences)
            id_list = torch.tensor(id_list[0]).to(src.device)
            
            #re-encoding summ_sentences to measure diversity
            embeddings_rencode = self.embedding(id_list.unsqueeze(-1))
            encoder_sum, hidden_sum = self.encoder(embeddings_rencode, src_max_len, rencode=rencode)
            hidden_sums.append(hidden_sum)
        
        hidden_sums_ = torch.stack(hidden_sums)
        hiddens = (hidden_sums_, hidden)
        
        return topic_summs, hiddens