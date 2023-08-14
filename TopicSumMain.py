#Main libraries
import pandas as pd
import os
from tqdm import tqdm
import nltk
import spacy
import math
import time
import sys
import random
import pickle

#torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

#torchtext libraries
from torchtext import data
from torchtext.vocab import GloVe

sys.path.append("configs")
import config as config

sys.path.append("utils")
from general_utils import *

#Personnalized libraries
sys.path.append("data")
from DataLoader import build_dataloader, load_create_extended

sys.path.append("experiments")
from Procedures import Procedure
from evaluation.Compute_evals import *

sys.path.append("model")
from model.main_model import VAEMultiSumm

import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Topic Oriented Summarization using VAE")

################ files ################
parser.add_argument("--root", default=config.path["root"], help="ROOT Directory where model is located")
parser.add_argument("--tota_vocab", default=config.path["tota_vocab"], help="File path to train extended vocabulary -- compiled file")
parser.add_argument("--extended_vocab", default=config.path["extended_vocab"], help="Path for pretrain extended vocabulary")
parser.add_argument("--train", default=config.path["train"], help="Path to training file")
parser.add_argument("--valid", default=config.path["valid"], help="Path to validation file")
parser.add_argument("--test", default=config.path["test"], help="Path to test file")
parser.add_argument("--vocab_path", default=config.path["vocab_path"], help="Path to test file")

################ data ################
parser.add_argument("--vocab_size", default=config.data["vocab_size"], type=int, help="Max size of the training vocabulary")
parser.add_argument("--ext_vocab_size", default=config.data["ext_vocab_size"], type=int, help="Max size of the extended vocabulary -- must be bigger than vocab size for PGN")
parser.add_argument("--min_freq", default=config.data["min_freq"], type=int, help="Minimum term frequency to be accounted for in vocab")
parser.add_argument("--max_num_reviews", default=config.data["max_num_reviews"], type=int, help="Max number of reviews per batch")
parser.add_argument("--max_len_rev", default=config.data["max_len_rev"], type=int, help="Max number of token in a review")
parser.add_argument("--shuffle_revs", default=config.data["shuffle_revs"], type=boolean_string, help="Shuffle reviews of product -- to set to True if you don't want to batch per product")
parser.add_argument("--n_prod_shuf", default=config.data["n_prod_shuf"], type=int, help="Number of different products to shuffle to sample reviews")
parser.add_argument("--n_fus", default=config.data["n_fus"], type=int, help="Number of product to fuse to create bigger batchs")
parser.add_argument("--recons_test", default=config.data["recons_test"], type=boolean_string, help="Do you have references for evaluation")
parser.add_argument("--shuffle_batch", default=config.data["shuffle_batch"], type=boolean_string, help="Shuffling batch in Dataloader")
parser.add_argument("--ban_list", default=config.data["ban_list"], nargs='+', help="Filtering POS Tags for Bag of Words representations")
parser.add_argument("--raw_data", default=config.data["raw_data"], type=boolean_string, help="Set to tru if data have not been preprocessed using folder Preprocess_data")
parser.add_argument("--topic_encod", default=config.data["topic_encod"], help="representation type for Bag of Words - individual reviews or group")
parser.add_argument("--encode_type", default=config.data["encode_type"], help="representation type for Bag of Words - one hot or frequency")
parser.add_argument("--review_field", default=config.data["review_field"], help="Name of the review column in the csv input file")

################ model ################
parser.add_argument("--emb_dim", default=config.model["emb_dim"], type=int, help="Embedding dimensions from Glove and model")
parser.add_argument("--enc_dim", default=config.model["enc_dim"], type=int, help="Dimension of hidden layers of the encoder")
parser.add_argument("--dec_dim", default=config.model["dec_dim"], type=int, help="Dimension of hidden layers of the decoder")
parser.add_argument("--attn_dim", default=config.model["attn_dim"], type=int, help="Dimension hidden layers of the attention heads")
parser.add_argument("--c_dim", default=config.model["c_dim"], type=int, help="Dimension of the latent variable c")
parser.add_argument("--z_dim", default=config.model["z_dim"], type=int, help="Dimension of the latent variable z")
parser.add_argument("--num_topics", default=config.model["num_topics"], type=int, help="Dimension of the latent variable t")
parser.add_argument("--drop", default=config.model["drop"], type=float, help="Dropout ratio for all model")
parser.add_argument("--use_pretrained", default=config.model["use_pretrained"], type=boolean_string, help="Use Glove embeddings or randomly initialized embeddings")
parser.add_argument("--n_layers", default=config.model["n_layers"], type=int, help="Number of layers in Encoder")
parser.add_argument("--gen_cond", default=config.model["gen_cond"], help="If X we concat t and z, if Z we concat c and t, and if BOTH we do both concat")
parser.add_argument("--topic_prior", default=config.model["topic_prior"], help="Prior condition for dirichlet distribution of topics")
parser.add_argument("--BoW_train", default=config.model["BoW_train"], type=boolean_string, help="Do we train Topic Model with Bag of Words")
parser.add_argument("--Glove_name", default=config.model["Glove_name"], help="6B default -- Glove model")
parser.add_argument("--encod_bidir", default=config.model["encod_bidir"], type=boolean_string, help="Bidirectionnal encoding of text")
parser.add_argument("--topic_prior_mean", default=config.model["topic_prior_mean"], type=float, help="Topic prior Mean")
parser.add_argument("--topic_prior_variance", default=config.model["topic_prior_variance"], type=float, help="Topic prior Variance")
parser.add_argument("--add_extrafeat", default=config.model["add_extrafeat"], type=boolean_string, help="Do we add extra features (z mean) in RNN decoding")
parser.add_argument("--add_extrafeat_pgn", default=config.model["add_extrafeat_pgn"], type=boolean_string, help="Do we add extra features (z mean) in PGN")
parser.add_argument("--context_vector_input", default=config.model["context_vector_input"], type=boolean_string, help="Do we use contextual attention vector in RNN preds")
parser.add_argument("--use_topic_attention", default=config.model["use_topic_attention"], type=boolean_string, help="Do we use topic dicstribution to bias decoder attention")

################ experiments ################
parser.add_argument("--writer", default=config.experiments["writer"], type=boolean_string, help="To keep track of losses in Tensorboard")
parser.add_argument("--Name", default=config.experiments["Name"], help="Name for saving the .pt trained model")
parser.add_argument("--write_rec_loss", default=config.experiments["write_rec_loss"], type=boolean_string, help="To keep track of reconstruction loss in Tensorboard")
parser.add_argument("--write_kl_loss", default=config.experiments["write_kl_loss"], type=boolean_string, help="To keep track of KL div loss in Tensorboard")
parser.add_argument("--write_BoW_loss", default=config.experiments["write_BoW_loss"], type=boolean_string, help="To keep track of Bag of Words loss in Tensorboard")
parser.add_argument("--write_total_loss", default=config.experiments["write_total_loss"], type=boolean_string, help="To keep track of the sum of all losses in Tensorboard")
parser.add_argument("--generate", default=config.experiments["generate"], type=boolean_string, help="If True, generate sumaries in json file")
parser.add_argument("--compute_evals", default=config.experiments["compute_evals"], type=boolean_string, help="Create csv with evaluation metrics")
parser.add_argument("--save_name", default=config.experiments["save_name"], help="Name for the different files saved")
parser.add_argument("--remove_stopwords", default=config.experiments["remove_stopwords"], type=boolean_string, help="Filtering stopwords for ROUGE eval")
parser.add_argument("--use_stemmer", default=config.experiments["use_stemmer"], type=boolean_string, help="Using Stemming for eval in ROUGE")
parser.add_argument("--train_model", default=config.experiments["train_model"], type=boolean_string, help="If True: Launch Training & Validation of the model")

################ Training params ################
parser.add_argument("--optimizer", default=config.train_params["optimizer"], help="Type of pytorch optimizer used")
parser.add_argument("--clip", default=config.train_params["clip"], type=float, help="Gradient clipping value")
parser.add_argument("--lr", default=config.train_params["lr"], type=float, help="Learning rate value")
parser.add_argument("--weight_decay", default=config.train_params["weight_decay"], type=float, help="Weight decay value")
parser.add_argument("--accumulation_steps", default=config.train_params["accumulation_steps"], type=int, help="Mini batch size accumulation")
parser.add_argument("--tf", default=config.train_params["tf"], type=float, help="Teacher Forcing value")
parser.add_argument("--N_epoch", default=config.train_params["N_epoch"], type=int, help="Total number of epochs for training the model")
parser.add_argument("--z_weight_max", default=config.train_params["z_weight_max"], type=float, help="KL div ponderation for Z in loss")
parser.add_argument("--c_weight_max", default=config.train_params["c_weight_max"], type=float, help="KL div ponderation for C in loss")
parser.add_argument("--t_weight_max", default=config.train_params["t_weight_max"], type=float, help="KL div ponderation for T in loss")
parser.add_argument("--n_epoch_cycle", default=config.train_params["n_epoch_cycle"], type=int, help="Cycling annealing param - what epoch to restart cycle")
parser.add_argument("--cycle_iter", default=config.train_params["cycle_iter"], type=int, help="Cycling annealing param - iteration for cycle")
parser.add_argument("--n_cycles_ct", default=config.train_params["n_cycles_ct"], type=int, help="Cycling annealing param - number of cycles")
parser.add_argument("--start_lin_t", default=config.train_params["start_lin_t"], type=int, help="Linear annealing - starting epoch")
parser.add_argument("--duration_lin_t", default=config.train_params["duration_lin_t"], type=int, help="Linear annealing - duration in epochs")
parser.add_argument("--momentum", default=config.train_params["momentum"], type=float, help="Momentum for optimizer")

################ Inference params ################
parser.add_argument("--beam_size", default=config.inference["beam_size"], type=int, help="Beam Size in Beam Search")
parser.add_argument("--min_steps", default=config.inference["min_steps"], type=int, help="Minimum of decoding steps to perform in beam search")
parser.add_argument("--num_return_seq", default=config.inference["num_return_seq"], type=int, help="Number of sequence to return for the Beam Search")
parser.add_argument("--num_return_sum", default=config.inference["num_return_sum"], type=int, help="Number of summary preserved in Beam Search")
parser.add_argument("--n_gram_block", default=config.inference["n_gram_block"], type=int, help="N-gram blocking size")
parser.add_argument("--gen_len_method", default=config.inference["gen_len_method"], help="Generation Method: NSumm or SingleSumm")
parser.add_argument("--topic_gen", default=config.inference["topic_gen"], help="Topic selection Method: select_main or word_bias")
parser.add_argument("--gen_word", default=config.inference["gen_word"], nargs='+', help="List of words to bias the summary generation")
parser.add_argument("--num_t_gen", default=config.inference["num_t_gen"], type=int, help="number of topics to generate")
parser.add_argument("--t_mask_len", default=config.inference["t_mask_len"], type=int, help="Size of the masking for topic tokens")
parser.add_argument("--rencode", default=config.inference["rencode"], type=boolean_string, help="Re-encoding summary for diveristy estimation")

args = parser.parse_args()    

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"The device you are using is {device}")
logger.info(args.vocab_path)
if args.train_model:
    logger.info(f"Loading dataloader for train")
    
    #Asserting functioning condition for building external vocabulary
    try:
        os.path.isfile(args.extended_vocab)
    except :
        try:
            os.path.isfile(args.tota_vocab)
        except FileError:
            logger.info("Please Provide at least a valid vocabulary or file to build the vocabulary")
    
    #Building external vocabulary and saving it for later - used 
    ext_vocab = load_create_extended(file_path=args.tota_vocab, vocab_size=args.ext_vocab_size, vocab_min_freq=args.min_freq, vocab_ext_path=args.extended_vocab, raw_data=args.raw_data)
    
    # instanciate dataloader for train and valid dataset
    train_iter, vocab, _ = build_dataloader(file_path=args.train, vocab_size=args.vocab_size, vocab_min_freq=args.min_freq, vocab=ext_vocab, max_num_reviews=args.max_num_reviews, max_len_rev=args.max_len_rev, is_train=True, shuffle_revs=args.shuffle_revs, n_prod_shuf=args.n_prod_shuf, n_fus= args.n_fus, shuffle_batch=args.shuffle_batch,  recons=False, device=device, ban_list=args.ban_list, raw_data=args.raw_data, vocab_path=args.vocab_path)
    train_size = len(train_iter)
    
    valid_iter, _, _ = build_dataloader(file_path=args.valid, vocab_size=args.vocab_size, vocab_min_freq=args.min_freq, vocab=ext_vocab,  vocab_train_size=len(vocab), max_num_reviews=args.max_num_reviews, max_len_rev=args.max_len_rev, is_train=False, shuffle_revs=args.shuffle_revs, n_prod_shuf=args.n_prod_shuf, n_fus=args.n_fus, shuffle_batch=args.shuffle_batch, recons=False, device=device, ban_list=args.ban_list, raw_data=args.raw_data)
    val_size = len(valid_iter)
    
if args.generate:
    
    #Making sure extended vocabularies has been loaded for the test
    try:
        ext_vocab
    except:
        ext_vocab = load_create_extended(file_path=args.tota_vocab, vocab_size=args.ext_vocab_size, vocab_min_freq=args.min_freq, vocab_ext_path=args.extended_vocab, raw_data=args.raw_data)
        #Loading trained vocab if only generation
        if not args.train_model:
            vocab_file = open(args.vocab_path, "rb")
            vocab = pickle.load(vocab_file)
            vocab_file.close()
    
    test_iter, _, test_references = build_dataloader(file_path=args.test, vocab_size=args.vocab_size, vocab_min_freq=args.min_freq, vocab=ext_vocab, max_num_reviews=args.max_num_reviews, max_len_rev=args.max_len_rev, is_train=False, shuffle_revs=args.shuffle_revs, n_prod_shuf=args.n_prod_shuf, n_fus= args.n_fus, shuffle_batch=args.shuffle_batch, recons=args.recons_test, device=device, ban_list=args.ban_list, raw_data=args.raw_data)
    test_size = len(test_iter)

#######################
######## Model ########
test_ = 1.-(1./args.num_topics)  
if test_ != args.topic_prior_variance:
    logger.info("It is not mandatory but it is weird that the topic variance is not the same as the number of topics")


model = VAEMultiSumm(ext_vocab, vocab, emb_dim=args.emb_dim, enc_dim=args.enc_dim, dec_dim=args.dec_dim, attn_dim=args.attn_dim, c_dim=args.c_dim, z_dim=args.z_dim, num_topics=args.num_topics, dropout=args.drop, device=device, topic_prior_mean=args.topic_prior_mean, topic_prior_variance=args.topic_prior_variance,  encod_bidir=args.encod_bidir, use_pretrained=args.use_pretrained, gen_cond=args.gen_cond, BoW_train=args.BoW_train,  Glove_name=args.Glove_name, Glove_dim=args.emb_dim,  beam_size=args.beam_size, min_dec_steps=args.min_steps, num_return_seq=args.num_return_seq, num_return_sum=args.num_return_sum,  n_gram_block=args.n_gram_block, add_extrafeat=args.add_extrafeat, add_extrafeat_pgn=args.add_extrafeat_pgn, context_vector_input=args.context_vector_input, use_topic_attention=args.use_topic_attention)

model.apply(xavier_weights_init)
model.to(device)
logger.info(f'The model has {count_parameters(model):,} trainable parameters')


if args.writer: 
    writer = SummaryWriter(comment=args.Name)
else:
    writer=None

procedure = Procedure(model, ext_vocab, vocab, writer=writer, n_epoch_cycle=args.n_epoch_cycle, optimizer=args.optimizer, clip=args.clip, learning_rate=args.lr, weight_decay=args.weight_decay, accumulation_steps=args.accumulation_steps, write_rec_loss=args.write_rec_loss, write_kl_loss=args.write_kl_loss, write_BoW_loss=args.write_BoW_loss, write_total_loss=args.write_total_loss, z_weight_max=args.z_weight_max, c_weight_max=args.c_weight_max, t_weight_max=args.t_weight_max, topic_encod=args.topic_encod, encode_type=args.encode_type, momentum=args.momentum, teacher_forcing_ratio=args.tf)

if args.train_model:
    
    logger.info(f"The model will be trained for {args.N_epoch} epochs")
    best_valid_loss = float('inf')
    start_time = time.perf_counter()
    
    for epoch in range(args.N_epoch):
    
        start_time = time.time()
        train_loss = procedure.train(train_iter, train_size, epoch, args.cycle_iter, n_cycles_ct=args.n_cycles_ct, start_lin_t=args.start_lin_t, duration_lin_t=args.duration_lin_t)
        if args.compute_evals:
            assert args.save_name is not None, "You need to provide a valid File name to evaluate (str)"
            valid_loss = procedure.evaluation(valid_iter, val_size, epoch, args.cycle_iter, references=None, save_name=args.save_name, start_lin_t=args.start_lin_t, duration_lin_t=args.duration_lin_t, gen_len_method=args.gen_len_method, topic_gen=args.topic_gen, gen_word=args.gen_word, n_cycles_ct=args.n_cycles_ct, num_t_gen=args.num_t_gen, t_mask_len=args.t_mask_len, rencode=args.rencode)
        else:
            valid_loss = procedure.evaluation(valid_iter, val_size, epoch, args.cycle_iter, references=None, save_name=None, start_lin_t=args.start_lin_t, duration_lin_t=args.duration_lin_t, gen_len_method=args.gen_len_method, topic_gen=args.topic_gen, gen_word=args.gen_word, n_cycles_ct=args.n_cycles_ct, num_t_gen=args.num_t_gen, t_mask_len=args.t_mask_len, rencode=args.rencode)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        ############## TEMP just to save model after one epoch for tests
        torch.save(model.state_dict(), f'experiments/model_save/{args.Name}.pt')
        # A changer pour save model pour le best ROUGE_validation aussi plutôt que juste la loss
        if epoch > 5:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f'experiments/model_save/{args.Name}.pt')
                
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\t Train Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')
        
        
###################################################
######## Generating Summary for test model ########
if args.generate: 
    if not args.train_model:
        procedure.model.load_state_dict(torch.load(f'experiments/model_save/{args.Name}.pt'))
        
    logger.info(f"The model is generating summaries")
    
    #Geenrating the json file for the test dataset
    procedure.generation(test_iter, test_size, epoch=0, cycle_iter=0, references=test_references, gen_len_method=args.gen_len_method, topic_gen=args.topic_gen, gen_word=args.gen_word, save_name=args.save_name, num_t_gen=args.num_t_gen, t_mask_len=args.t_mask_len, rencode=args.rencode)
    
###################################################
######## Computing evaluation for validation and test files ########

if args.compute_evals:
    
    if os.path.isfile(f'experiments/results/results_{args.save_name}_valid.json'):
        logger.info(f"Computing evaluation metrics for the validation set")
        results_path = f'experiments/results/results_{args.save_name}_valid.json'
        _pt_path = f'experiments/results/hidden_{args.save_name}_valid.pt'
        compute = compte_evals(results_path, gen_method=args.gen_len_method)
        bleurt_dict = compute.compute_BLEURT()
        diversity_ = diversity(_pt_path)
        compute_TM = eval_TM(results_path, gen_method=args.gen_len_method, train_path=args.train, review_field=args.review_field, num_k=args.num_topics, ban_list=args.ban_list)
        vector_sums, vector_batch_reviews, _ = compute_TM.compute()
        count_best_match, average_sim_overlap, _ = compute_TM.best_topic(vector_sums, vector_batch_reviews, top=100, n_gens=args.num_t_gen)
        avg_topic_sim = compute_TM.sim_topics(vector_sums, vector_batch_reviews, n_gens=args.num_t_gen)
        
        #Papers indicated metrics
        dict_final = {"BELURT": bleurt_dict["final_avg_revs"], "Word_Overlap": average_sim_overlap, "Topic_similarity":avg_topic_sim}
        #save in pandas csv
        df = pd.DataFrame(dict_final, index=[0])
        df.to_csv(f"experiments/results/{args.save_name}_metrics_valid.csv", index=False)
        
    if os.path.isfile(f'experiments/results/results_{args.save_name}_gen.json'):
        logger.info(f"Computing evaluation metrics for the test set")
        results_path = f'experiments/results/results_{args.save_name}_gen.json'
        _pt_path = f'experiments/results/hidden_{args.save_name}_gen.pt'
        compute = compte_evals(results_path, gen_method=args.gen_len_method)
        avg_rouge_dict, max_rouge_dict = compute.compute_rouge()
        filt_avg_rouge_dict, filt_max_rouge_dict = compute.compute_rouge(remove_stopwords=args.remove_stopwords, use_stemmer= args.use_stemmer)
        bleurt_dict = compute.compute_BLEURT()
        frag_dict = compute.frag_coverage()
        dens_dict = compute.density
        diversity_ = diversity(_pt_path)
        compute_TM = eval_TM(results_path, gen_method=args.gen_len_method, train_path=args.train, review_field=args.review_field, num_k=args.num_topics, ban_list=args.ban_list)
        vector_sums, vector_batch_reviews, _ = compute_TM.compute()
        count_best_match, average_sim_overlap, _ = compute_TM.best_topic(vector_sums, vector_batch_reviews, top=100, n_gens=args.num_t_gen)
        avg_topic_sim = compute_TM.sim_topics(vector_sums, vector_batch_reviews, n_gens=args.num_t_gen)
    
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
        df.to_csv(f"experiments/results/{args.save_name}_metricsGen.csv", index=False)