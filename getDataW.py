

from torch.autograd import Variable
import numpy as np
import os
import torch

import mf_utils as util

num_fasc = 2

use_pretrained = False
use_GPU = False

sum_to_one = False
remove_mean = True
unit_variance = True
center_target = True
SNR_dist = 'uniform'  # 'uniform' or 'triangular'
normalisation = False

num_epochs = 5
batch_size = 200
myseed = 141414

target_names = ['nu1', 'r1 ', 'f1 ', 'nu2', 'r2 ', 'f2 ']
num_var = len(target_names)

assert unit_variance != sum_to_one, ("Choose one of two normalization "
                                     "strategies: either unit sum or unit "
                                     "variance.")

#Test    
if SNR_dist == 'uniform':
    nnls_output = util.loadmat('data_TEST3_article/training_data_fixedSNR_15000_samples_lou_TEST3_article')
    validation_data = nnls_output
    print(nnls_output['nnz_hist_0'])

# Substrate (=fingerprint) properties
sub_rads = nnls_output['subinfo']['rad']  # Python list
sub_fins = nnls_output['subinfo']['fin']  # Python list
tot_samples = nnls_output['num_samples']

#%% For normalization of target outputs

if normalisation:
    rad_min = np.min(nnls_output['subinfo']['rad'])
    rad_range = np.max(nnls_output['subinfo']['rad']) - rad_min
    print(rad_range)
    fin_min = np.min(nnls_output['subinfo']['fin'])
    fin_range = np.max(nnls_output['subinfo']['fin']) - fin_min
    nu_min = nnls_output['nu_min']
    nu_range = nnls_output['nu_max'] - nu_min
    trgt_true_min = np.array([nu_min, rad_min, fin_min,
                              nu_min, rad_min, fin_min])
    trgt_true_range = np.array([nu_range, rad_range, fin_range,
                                nu_range, rad_range, fin_range])
    if center_target:
    # Ultimately, we should map each variable to [-sqrt(3), sqrt(3)] to ensure
    # unit variance of the output
        trgt_proj_min = np.array([-nu_range/2, -0.5, -fin_range/2,
                                  -nu_range/2, -0.5, -fin_range/2])
        trgt_proj_range = np.array([nu_range, 1, fin_range,
                                    nu_range, 1, fin_range])
    else:
    # Initially, we let nu and fin take their "natural" values and mapped rad
    # to [0,1]
        trgt_proj_min = np.array([nu_min, 0, fin_min,
                                  nu_min, 0, fin_min])
        trgt_proj_range = np.array([nu_range, 1, fin_range,
                                    nu_range, 1, fin_range])
 
    

#%% Function

def gen_batch_data(start, end, mode):
    print(mode)
    if end > tot_samples:
        raise ValueError("Only %d data samples available. Asked for samples "
                         "up to %d." % (tot_samples, end))
    batch_size = end-start

    if mode == 'train':
        print('avec est ori')
        if nnls_output['sparse']:
            w_store = np.zeros((batch_size,
                                nnls_output['num_fasc'] *
                                nnls_output['num_atoms']))
            isbatch = ((nnls_output['w_idx'][:, 0] >= start) &
                       (nnls_output['w_idx'][:, 0] < end))
            chk = (np.sum(isbatch) ==
                   np.sum(nnls_output['nnz_hist'][start:end]))
            assert chk, ("Mismatch non-zero elements in samples "
                         "%d (incl.) to %d (excl.)" % (start, end))
            w_idx = nnls_output['w_idx'][isbatch, :]
            w_store[w_idx[:, 0] - start,
                    w_idx[:, 1]] = nnls_output['w_data'][isbatch]
        else:
            w_store = nnls_output['w_store'][start:end, :]
    
    elif mode == 'TrueOri':
        print('avec true ori')
        if nnls_output['sparse']:
            print('sparse')
            w_store = np.zeros((batch_size,
                                nnls_output['num_fasc'] *
                                nnls_output['num_atoms']))
            isbatch = ((nnls_output['w_idx_0'][:, 0] >= start) &
                       (nnls_output['w_idx_0'][:, 0] < end))
            chk = (np.sum(isbatch) ==
                   np.sum(nnls_output['nnz_hist_0'][start:end]))
            assert chk, ("Mismatch non-zero elements in samples "
                         "%d (incl.) to %d (excl.)" % (start, end))
            w_idx = nnls_output['w_idx_0'][isbatch, :]
            w_store[w_idx[:, 0] - start,
                    w_idx[:, 1]] = nnls_output['w_data_0'][isbatch]
        else:
            w_store = nnls_output['w_store_0'][start:end, :]

    #else:
        #raise ValueError('Unknown mode %s' % mode
        
    # NNLS weights no more normalized to sum to 1 after April 26, 2019.
    if sum_to_one:
        # Must be done before mean is removed!
        w_store = w_store/np.sum(w_store, axis=1)[:, np.newaxis]

    if remove_mean:
        w_store = w_store - np.mean(w_store, axis=1)[:, np.newaxis]

    if unit_variance:
        std_w = np.std(w_store, axis=1)
        idx_pos_std = np.where(std_w > 0)[0]
        w_store[idx_pos_std, :] = (w_store[idx_pos_std, :] /
                                   std_w[idx_pos_std][:, np.newaxis])
        # Zero variance (constant weights): normalize if non-zero weights
        # Case which should not occur too often: w_store[i, j] = C > 0 for all
        # j in [0, num_fasc*num_atoms]
        w_L1 = np.sum(np.abs(w_store), axis=1)  # (Nbatch,)
        idx_pos_const = np.where((std_w == 0) & (w_L1 > 0))[0]
        if idx_pos_const.size > 0:
            w_store[idx_pos_const, :] = (w_store[idx_pos_const, :] /
                                         w_L1[idx_pos_const][:, np.newaxis])
            print("%d samples containing identical positive weights for "
                  "all the atoms of the dictionary!" % idx_pos_const.size)

    # Data contains w12 normalized to sum to one
    data = torch.from_numpy(w_store).float()

    # Target contains nu1, r1, f1, nu2, r2, f2
    batch_IDs = nnls_output['IDs'][start:end, :]
    batch_nus = nnls_output['nus'][start:end, :]

    
    #Normalisation ou pas
    target = torch.FloatTensor(batch_size, num_fasc * (1 + 2)).zero_()
    if normalisation == True:
        target[:, [0, 3]] = (trgt_proj_min[0] +
                             trgt_proj_range[0] *
                             (torch.from_numpy(batch_nus).float() -
                              trgt_true_min[0]) /
                             trgt_true_range[0])
        target[:, [1, 4]] = (trgt_proj_min[1] +
                             trgt_proj_range[1] *
                             (torch.FloatTensor(sub_rads)[batch_IDs] -
                              trgt_true_min[1]) /
                             trgt_true_range[1])
        target[:, [2, 5]] = (trgt_proj_min[2] +
                             trgt_proj_range[2] *
                             (torch.FloatTensor(sub_fins)[batch_IDs] -
                              trgt_true_min[2]) /
                             trgt_true_range[2])
    elif normalisation == False:
        target[:, [0, 3]] = torch.from_numpy(batch_nus).float()
        target[:, [1, 4]] = torch.FloatTensor(sub_rads)[batch_IDs]
        target[:, [2, 5]] = torch.FloatTensor(sub_fins)[batch_IDs]
        
    if use_GPU and torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()
    return Variable(data), Variable(target)