# -*- coding: utf-8 -*-
"""
Created on 11/04/2021

Create synthetic data for test with a given protocol 

@author: louiseadam
"""

import sys
import os
import numpy as np
import socket
from math import pi
import time
import scipy.io as scio
import matplotlib.pyplot as plt

# Load mf_utils
path_to_utils = os.path.join('python_functions')
path_to_utils = os.path.abspath(path_to_utils)
if path_to_utils not in sys.path:
    sys.path.insert(0, path_to_utils)
import mf_utils as util

# ---- Set input parameters here -----
#use_prerot = False  # use pre-rotated dictionaries
#sparse = True # store data sparsely to save space
#save_res = False  # save mat file containing data
#num_samples = 1 #originally at 100
#save_dir = 'synthetic_data'  # destination folder

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


# %% Load single-fascicle canonical dictionary
ld_singfasc_dic = util.loadmat('MC_dictionary_hcp.mat')

dic_sing_fasc = np.zeros(ld_singfasc_dic['dic_fascicle_refdir'].shape)
dic_sing_fasc[ind_b0,:] = ld_singfasc_dic['dic_fascicle_refdir'][:num_B0, :]
dic_sing_fasc[ind_b,:] = ld_singfasc_dic['dic_fascicle_refdir'][num_B0:, :]
refdir = np.array([0.0, 0.0, 1.0])

# Paramètres du protocole
num_atoms = ld_singfasc_dic['dic_fascicle_refdir'].shape[1]
WM_DIFF = ld_singfasc_dic['WM_DIFF']
S0_fasc = ld_singfasc_dic['S0_fascicle']
sig_csf = ld_singfasc_dic['sig_csf']  # already T2-weighted as well
subinfo = ld_singfasc_dic['subinfo']  # just used for displaying results

S0_max = np.max(S0_fasc)
assert num_atoms == len(subinfo['rad']), "Inconsistency dictionaries"

# %% Generate synthetic acquisition
M0 = 500
num_fasc = 2
nu_min = 0.15
nu_max = 1 - nu_min
#SNR_min = 80
#SNR_max = 100
#SNR_max = 30
num_coils = 1
crossangle_min = 15 * pi/180
cos_min = np.cos(crossangle_min)

def gen_test_data(num_samples, use_noise=False, SNR_min=80, SNR_max=100):
    
    #SNR_min = 80
    #SNR_max = 100
    SNR_dist = 'uniform'  # 'uniform' or 'triangular'
    starttime = time.time()
    
    # Prepare memory
    IDs = np.zeros((num_samples, num_fasc), dtype=np.int32)
    nus = np.zeros((num_samples, num_fasc))
    SNRs = np.zeros(num_samples)
    
    DW_image_store = np.zeros((552, num_samples))
    DW_noisy_store = np.zeros((552, num_samples))
    
    orientations = np.zeros((num_samples, num_fasc, 3))
    
    dic_compile = np.zeros((num_samples, num_mris, num_fasc * num_atoms), dtype=np.float64)
    
    dictionary = np.zeros((num_mris, num_fasc * num_atoms), dtype=np.float64)
    
    dictionary[:, :num_atoms] = dic_sing_fasc  #first direction fixed
    
    for i in range(num_samples):
        if i % 100 ==0:
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
        
        dic_sing_fasc_2 = util.rotate_atom(dic_sing_fasc,
                                           sch_mat_b0, refdir, cyldir_2,
                                           WM_DIFF, S0_fasc)
        dictionary[:, num_atoms:] = dic_sing_fasc_2
        
        dic_compile[i, :, :] = dictionary
        
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
    
    return DW_image_store, DW_noisy_store, dic_compile, time_elapsed, sch_mat_b0, IDs, nus
