# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 18:12:07 2019

Pipeline to create synthetic data with a given protocol lending itself to
easy rotations (here the HCP-MGH multi-HARDI protocol) in a form suitable for
training a (moderately) deep network.

The script also performs the initial NNLS estimation and stores the estimated
weights, possibly in a compact (sparse) form to save disk space.

April 26, 2019: removed normalization to unit sum of NNLS weights

Benchmarking on Rastaban CentOS 7 with prerotated dictionaries:
    10*6 samples: 6734.71s (=1h52m14.71s = 112m14.71s)
Quite longer without using prerotated dictionaries.

@author: rensonnetg
"""

import sys
import os
import numpy as np
import socket
from math import pi
import time
import scipy.io as scio

import matplotlib.pyplot as plt

# See check_synthetic_data.py for explanations of the lines below
path_to_utils = os.path.join('python_functions')
path_to_utils = os.path.abspath(path_to_utils)
if path_to_utils not in sys.path:
    sys.path.insert(0, path_to_utils)
import mf_utils as util

# ---- Set input parameters here -----
use_prerot = False  # use pre-rotated dictionaries
sparse = True # store data sparsely to save space
save_res = False  # save mat file containing data
SNR_dist = 'uniform'  # 'uniform' or 'triangular'
num_samples = 10 #originally at 100
save_dir = 'synthetic_data'  # destination folder

# Initiate random number generator (to make results reproducible)
rand_seed = 141414
np.random.seed(rand_seed)

# %% Load DW-MRI protocol from Human Connectome Project (HCP)
schemefile = os.path.join('real_data', 'hcp_mgh_1003.scheme1')
sch_mat = np.loadtxt(schemefile, skiprows=1)  # only DWI, no b0s
bvalfile = os.path.join('real_data', 'bvals.txt')
bvals = np.loadtxt(bvalfile)
ind_b0 = np.where(bvals <= 1e-16)[0]
ind_b = np.where(bvals > 1e-16)[0]
num_B0 = ind_b0.size
sch_mat_b0 = np.zeros((sch_mat.shape[0] + num_B0, sch_mat.shape[1]))
sch_mat_b0[ind_b0, 4:] = sch_mat[0, 4:]
sch_mat_b0[ind_b, :] = sch_mat
num_mris = sch_mat_b0.shape[0]

print('ind_b0', ind_b0)
print('ind_b', ind_b)
print('num_B0', num_B0)
print('sch_mat_b0', sch_mat_b0)
print('num_mris', num_mris)

# %% Load single-fascicle canonical dictionary
ld_singfasc_dic = util.loadmat('MC_dictionary_hcp.mat')
# The single-fascicle dictionary stored in the matfile contains all the b0
# images first followed by the diffusion-weighted signals. We reorder the
# acquisitions to match the acquisition protocol.
if not use_prerot:
    dic_sing_fasc = np.zeros(ld_singfasc_dic['dic_fascicle_refdir'].shape)
    dic_sing_fasc[ind_b0,
                  :] = ld_singfasc_dic['dic_fascicle_refdir'][:num_B0, :]
    dic_sing_fasc[ind_b,
                  :] = ld_singfasc_dic['dic_fascicle_refdir'][num_B0:, :]
    refdir = np.array([0.0, 0.0, 1.0])

# Paramètres du protocole
num_atoms = ld_singfasc_dic['dic_fascicle_refdir'].shape[1]
WM_DIFF = ld_singfasc_dic['WM_DIFF']
S0_fasc = ld_singfasc_dic['S0_fascicle']
sig_csf = ld_singfasc_dic['sig_csf']  # already T2-weighted as well
subinfo = ld_singfasc_dic['subinfo']  # just used for displaying results

print('num_atoms', num_atoms)
print('WM_DIFF', WM_DIFF)
print('S0_fasc', S0_fasc.shape)
print('sig_csf', sig_csf.shape)
#print('subinfo', subinfo)

S0_max = np.max(S0_fasc)
assert num_atoms == len(subinfo['rad']), "Inconsistency dictionaries"

# %% Generate synthetic acquisition
M0 = 500
num_fasc = 2
nu_min = 0.15
nu_max = 1 - nu_min
SNR_min = 80
SNR_max = 100
#SNR_max = 30
num_coils = 1
crossangle_min = 15 * pi/180
cos_min = np.cos(crossangle_min)

# Estimate RAM requirements
RAM_dense = (num_fasc*num_atoms*num_samples*8  # NNLS weights
             + num_fasc*num_samples*4  # IDs
             + num_fasc*num_samples*8  # nus
             + num_fasc*num_samples*(3-2*use_prerot)*8  # orientations
             + num_samples*8  # SNRs
             )
if RAM_dense > 1e9 and not sparse:
    raise ValueError("At least %5.4f Gb of RAM required. "
                     "Sparse mode should be used." % RAM_dense/1e9)

starttime = time.time()
time_rot_hist = np.zeros(num_samples)

# Prepare memory
IDs = np.zeros((num_samples, num_fasc), dtype=np.int32)
nus = np.zeros((num_samples, num_fasc))
SNRs = np.zeros(num_samples)

DW_image_store = np.zeros((552, num_samples))
DW_noisy_store = np.zeros((552, num_samples))

orientations = np.zeros((num_samples, num_fasc, 3))
if sparse:
    sparsity = 0.01  # expected proportion of nnz atom weights per fascicle
    nnz_pred = int(np.ceil(sparsity * num_atoms * num_samples * num_fasc))
     # Store row and column indices of the dense weight matrix
    w_idx = np.zeros((nnz_pred, 2), dtype=np.int64)  # 2 is 2 !
    # Store weights themselves
    w_data = np.zeros(nnz_pred, dtype=np.float64)
else:
    w_store = np.zeros((num_samples, num_fasc*num_atoms), dtype=np.float)

nnz_hist = np.zeros(num_samples)  # always useful even in non sparse mode

dictionary = np.zeros((num_mris, num_fasc * num_atoms), dtype=np.float64)
if not use_prerot:
    dictionary[:, :num_atoms] = dic_sing_fasc  # in case first direction fixed

nnz_cnt = 0  # non-zero entries (just for sparse case)

for i in range(num_samples):
    if i % 1000 ==0:
        print(i)
    
    nu1 = nu_min + (nu_max - nu_min) * np.random.rand()
    nu2 = 1 - nu1
    ID_1 = np.random.randint(0, num_atoms)
    ID_2 = np.random.randint(0, num_atoms)
    if SNR_dist == 'triangular':
        SNR = np.random.triangular(SNR_min, SNR_min, SNR_max, 1)
    elif SNR_dist == 'uniform':
        SNR = np.random.uniform(SNR_min, SNR_max, 1)
    else:
        raise ValueError("Unknown SNR distribution %s" % SNR_dist)

    sigma_g = S0_max/SNR

    # First fascicle direction fixed, second fascicle rotated on the fly
    cyldir_1 = refdir
    cyldir_2 = refdir.copy()
    while np.dot(refdir, cyldir_2) > np.cos(crossangle_min):
        cyldir_2 = np.random.randn(3)
        norm_2 = np.sqrt(np.sum(cyldir_2**2))
        if norm_2 < 1e-11:
            cyldir_2 = refdir
        else:
            cyldir_2 = cyldir_2/norm_2
    start_rot = time.time()
    dic_sing_fasc_2 = util.rotate_atom(dic_sing_fasc,
                                       sch_mat_b0, refdir, cyldir_2,
                                       WM_DIFF, S0_fasc)
    dictionary[:, num_atoms:] = dic_sing_fasc_2
    time_rot_hist[i] = time.time() - start_rot
    
    # Assemble synthetic DWI
    DW_image = (nu1 * dic_sing_fasc[:, ID_1]
                + nu2 * dic_sing_fasc_2[:, ID_2])
    
    # Simulate noise and MRI scanner scaling
    DW_image_store[:, i] = DW_image
    
    DW_image_noisy = util.gen_SoS_MRI(DW_image, sigma_g, num_coils)
    #DW_image_noisy = M0 * DW_image_noisy
    
    DW_noisy_store[:, i] = DW_image_noisy
    
    # Store
    IDs[i, :] = np.array([ID_1, ID_2])
    nus[i, :] = np.array([nu1, nu2])
    
    
time_elapsed = time.time() - starttime
#print('%d samples created in %g sec.' % (num_samples, time_elapsed))

#%% Save in files

mdict = {'rand_seed': rand_seed,
         'M0': M0,
         'num_fasc': num_fasc,
         'nu_min': nu_min,
         'nu_max': nu_max,
         'SNR_min': SNR_min,
         'SNR_max': SNR_max,
         'SNR_dist': SNR_dist,
         'num_coils': num_coils,
         'crossangle_min': crossangle_min,
         'num_fasc': num_fasc,
         'num_atoms': num_atoms,
         'WM_DIFF': WM_DIFF,
         'S0_fasc': S0_fasc,
         'S0_max': S0_max,
         'sig_csf': sig_csf,
         'subinfo': subinfo,
         'sch_mat_b0': sch_mat_b0,
         'num_samples': num_samples,
         'IDs': IDs,
         'nus': nus,
         'SNRs': SNRs,
         'nnz_hist': nnz_hist,
         'orientations': orientations,
         'use_prerot': use_prerot,
         }

if save_res:
    if SNR_dist == 'uniform':
        SNR_str = 'uniform'
    elif SNR_dist == 'triangular':
        SNR_str = '_triangSNR'
    else:
        raise ValueError('Unknown SNR distribution %s' % SNR_dist)
    fname = os.path.join(save_dir, "training_data%s_%d_samples_version2" %
                         (SNR_str, num_samples))
    scio.savemat(fname,
                 mdict,
                 format='5',
                 long_field_names=False,
                 oned_as='column')
    
#%% Pickle files

import pickle

filename = 'dw_image_data_version2'
with open(filename, 'wb') as f:
        pickle.dump(DW_image_store, f)
        f.close()

filename = 'dw_noisy_data_version2' 
with open(filename, 'wb') as f:
        pickle.dump(DW_noisy_store, f)
        f.close()
        
filename = 'ID_noisy_data_version2' 
with open(filename, 'wb') as f:
        pickle.dump(IDs, f)
        f.close()
        
filename = 'nus_data_version2' 
with open(filename, 'wb') as f:
        pickle.dump(nus, f)
        f.close()



