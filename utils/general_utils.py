from datetime import datetime
import numpy as np
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
            
def xavier_weights_init(m):
    for name, param in m.named_parameters():
        #Here len(param.shape)> 1 is to not initialize weigths for the self.pt_alphas = nn.Parameter(torch.ones((1, num_topics))) layer - which is needed to assume uniform dirichlet distribution
        if 'weight' in name and len(param.shape)> 1:
            nn.init.xavier_uniform_(param.data)
        else:
            if 'weight':
                pass
            else:
                nn.init.constant_(param.data, 0)
                
def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=20, ratio=0.8):
    """
    Cycling annealing function -- for t and c 
    Implemeted as --- Fu, H., et. al. (2019). Cyclical annealing schedule: A simple approach to mitigating kl vanishing.
    M: n_cycle - the number of cycles; and 2) R: ratio - the proportion used to increase β (and 1-R used to fix β)
    Default M at 2 epochs and r = 0.8
    """

    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 


def constant_annealing(epoch_size, epoch, iter_, var_weight,  start = 20, duration = 40):
    """
    Linear annealing function -- for topics
    args: 
        start : number of epoch where to start updating lambda weight
        duration : number of total epoch when update is done then kl_term is set to 1
    """
    
    kl_scheduler = epoch_size * epoch + iter_             #What step we are
    start_scheduler = epoch_size * start                  #What step we start update KL weight
    duration_scheduler = epoch_size * duration            #What setp we stop updating KL weight and use 1
    
    return max(min((kl_scheduler - start_scheduler) / duration_scheduler, var_weight), 0)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
