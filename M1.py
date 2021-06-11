#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 21:19:29 2021

@author: louiseadam
"""

from math import pi
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as scio
import sys
import time

from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import mf_utils as util
import pickle
from matplotlib.colors import LogNorm


# ---- Set input parameters here -----
n_test = '11_6_TEST3_descaled' # used for saving

save_err = True # save error in pickle file

num_sample = 15000
M0 = 500
num_fasc = 2

# Initiate random number generator (to make results reproducible)
rand_seed = 141414
np.random.seed(rand_seed)

#%% Load Dw_image data

use_noise = True
use_NoNoise = False

if use_noise:
    filename = 'data_TEST3_article/DW_noisy_store__fixedSNR_15000__lou_TEST3_article'
    y_data = pickle.load(open(filename, 'rb'))
    y_data = y_data/M0
    print('ok noise')
    
if use_NoNoise:   
    filename = 'data_TEST3_article/DW_image_store__fixedSNR_15000__lou_TEST3_article'
    y_data2 = pickle.load(open(filename, 'rb'))    
    print('ok no noise')
    
target_data = util.loadmat(os.path.join('data_TEST3_article',
                                            "training_data_fixedSNR_15000_samples_lou_TEST3_article"))

IDs = target_data['IDs'][0:num_sample, :]
nus = target_data['nus'][0:num_sample, :]
    
target_params_y = np.zeros((6, num_sample))

target_params_y[0,:] = nus[:,0]
target_params_y[1,:] = target_data['subinfo']['rad'][IDs[:,0]]
target_params_y[2,:] = target_data['subinfo']['fin'][IDs[:,0]]
target_params_y[3,:] = nus[:,1]
target_params_y[4,:] = target_data['subinfo']['rad'][IDs[:,1]]
target_params_y[5,:] = target_data['subinfo']['fin'][IDs[:,1]]

scaled = False
if scaled:
    scaler_y = StandardScaler()
    target_params_y = scaler_y.fit_transform(target_params_y.T)
    target_params_y = target_params_y.T

baseline = np.mean(abs(target_params_y), 0)
mean_baseline_prop = np.mean(abs(target_params_y), 1)
 
        
# %% Load DW-MRI protocol from Human Connectome Project (HCP)
schemefile = os.path.join('real_data', 'hcp_mgh_1003.scheme1')
sch_mat = np.loadtxt(schemefile, skiprows=1)  # only DWI, no b0s
bvalfile = os.path.join('real_data', 'bvals.txt')
bvecfile = os.path.join('real_data', 'bvecs.txt')
bvals = np.loadtxt(bvalfile)  # NOT in SI units, in s/mm^2
ind_b0 = np.where(bvals <= 1e-16)[0]
ind_b = np.where(bvals > 1e-16)[0]
num_B0 = ind_b0.size
sch_mat_b0 = np.zeros((sch_mat.shape[0] + num_B0, sch_mat.shape[1]))
sch_mat_b0[ind_b0, 4:] = sch_mat[0, 4:]
sch_mat_b0[ind_b, :] = sch_mat
num_mris = sch_mat_b0.shape[0]

# %% Load single-fascicle canonical dictionary
ld_singfasc_dic = util.loadmat('MC_dictionary_hcp.mat')
# The single-fascicle dictionary stored in the matfile contains all the b0
# images first followed by the diffusion-weighted signals. We reorder the
# acquisitions to match the acquisition protocol. This is the canonical
# dictionary.
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

S0_max = np.max(S0_fasc)
assert num_atoms == len(subinfo['rad']), "Inconsistency dictionaries"

#%% 1 ORIENTATION ESTIMATION

cyldir_1 = target_data['orientations'][:, 0, :]
cyldir_2 = target_data['orientations'][:, 1, :]

est_dir1 = target_data['est_orientations'][:, 0, :]
est_dir2 = target_data['est_orientations'][:, 1, :]


#%% 2 ROTATION OF DICTIONARY
#   and
#   3 DICTIONNARY SEARCH

num_sample = 15000
w_nneg_store = np.zeros((num_sample,2))
ind_atoms_subdic_store = np.zeros((num_sample,2))
start_search = time.time()
dicsizes = np.array([782, 782])

for i in range(num_sample):
    
    if i%10==0:
            print("sample:", i)
    
    dictionary = np.zeros((num_mris, num_fasc * num_atoms), dtype=np.float64)
    dictionary[:, :num_atoms] = dic_sing_fasc
    dictionary[:, num_atoms:] = util.rotate_atom(dic_sing_fasc,
                                                 sch_mat_b0,
                                                 refdir,
                                                 #cyldir_2[i,:],
                                                 est_dir2[i,:],
                                                 WM_DIFF, S0_fasc)
    
    w_nneg, ind_atoms_subdic, ind_atoms_totdic, min_obj, y_recons = util.solve_exhaustive_posweights(dictionary, y_data[:, i], dicsizes)
    
    w_nneg_store[i, :] = w_nneg
    ind_atoms_subdic_store[i, :] = ind_atoms_subdic
    
time_search = time.time() - start_search   

#%% Calcul et enregistrement de l'erreur

est_params = np.zeros((6, num_sample))

indexes = ind_atoms_subdic_store.astype(int)

est_params[0,:] = w_nneg_store[:,0]
est_params[1,:] = subinfo['rad'][indexes[:,0]]
est_params[2,:] = subinfo['fin'][indexes[:,0]]
est_params[3,:] = w_nneg_store[:,1]
est_params[4,:] = subinfo['rad'][indexes[:,1]]
est_params[5,:] = subinfo['fin'][indexes[:,1]]

if scaled:
    est_params = scaler_y.transform(est_params.T)
    est_params = est_params.T

error = abs(est_params - target_params_y[:, :num_sample])

error_prop = np.mean(error, 1)

mean_error = np.mean(error_prop)
print(error_prop)

if save_err:
    filename = 'error_M1_testdata_EstOrientations_%s' %n_test
    with open(filename, 'wb') as f:
        pickle.dump(error, f)
        f.close()
        print("error saved")

#%% Evaluate estimation of orientations (code Gaetan)

orientations = target_data['orientations']

est_orientations = target_data['est_orientations']

dp_option1_1 = np.abs(np.sum(orientations[:, 0, :] * est_orientations[:, 0, :], axis=-1))
dp_option1_2 = np.abs(np.sum(orientations[:, 1, :] * est_orientations[:, 1, :], axis=-1))
dp_option2_1 = np.abs(np.sum(orientations[:, 0, :] * est_orientations[:, 1, :], axis=-1))
dp_option2_2 = np.abs(np.sum(orientations[:, 1, :] * est_orientations[:, 0, :], axis=-1))

# remove numerical round off errors, make sure results are in [0, 1]
dp_option1_1 = np.clip(dp_option1_1, 0, 1)
dp_option1_2 = np.clip(dp_option1_2, 0, 1)
dp_option2_1 = np.clip(dp_option2_1, 0, 1)
dp_option2_2 = np.clip(dp_option2_2, 0, 1)

dp_option1 = dp_option1_1 + dp_option1_2
dp_option2 = dp_option2_1 + dp_option2_2

is_option1 = dp_option1 >= dp_option2
# this implies that ~is_option1 corresponds to option 2

num_samples = orientations.shape[0]
err_dir1 = np.zeros(num_samples)  # error with respect to gt dir 1
err_dir1[is_option1] = np.arccos(dp_option1_1[is_option1]) * 180/np.pi
err_dir1[~is_option1] = np.arccos(dp_option2_1[~is_option1]) * 180/np.pi

err_dir2 = np.zeros(num_samples)  # error with respect to gt dir 2
err_dir2[is_option1] = np.arccos(dp_option1_2[is_option1]) * 180/np.pi
err_dir2[~is_option1] = np.arccos(dp_option2_2[~is_option1]) * 180/np.pi

dp_est = np.sum(est_orientations[:, 0, :]*est_orientations[:, 1, :], axis=-1)
est_ang_sep = np.arccos(np.clip(np.abs(dp_est), 0, 1)) * 180/np.pi

# error

mean_ang_err = (abs(err_dir1) + abs(err_dir2))/2
ang_err2 = np.zeros((2, num_sample))
ang_err2[0, :] = err_dir1 
ang_err2[1, :] = err_dir2 

# Diviser les ang_err en fonction du bruit et de nu
reshape_ang_err = np.zeros((3, 5, 1000))
reshape_ang_err2 = np.zeros((2, 3, 5, 1000))
for i in range(3):
    for j in range(5):
        elem = j*3 + i
        reshape_ang_err[i,j, :] = mean_ang_err[elem*1000:(elem+1)*1000]
        
        reshape_ang_err2[:, i,j, :] = ang_err2[:,elem*1000:(elem+1)*1000]
        

#%% Histogram of angular error

plt.hist(mean_ang_err, bins=15, color='steelblue', edgecolor='white')
plt.title('Histogram of angular error for orientation estimation')
plt.ylabel('Number of samples')
plt.xlabel('Agular error')
plt.grid(True, axis='y')

plt.savefig("graphs/AngErr_test%s.pdf" %n_test, dpi=150) 

#%% hist2D --> Fail 

ang_err = np.array([err_dir1, err_dir2])
plt.hist2d(abs(err_dir1), abs(err_dir2), bins=6, alpha=0.8, cmin = 0.9, norm=LogNorm())
plt.colorbar()
plt.title('Histogram of angular error for orientation estimation')
plt.ylabel('Number of samples')
plt.xlabel('Agular error')

plt.savefig("graphs/AngErr_Hist2d_test%s.pdf" %n_test, dpi=150) 

#%% subplots

fig1, ax1 = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
fig1.suptitle('Angular error for different noise levels')
nu_min = [0.5, 0.4, 0.3, 0.2, 0.1]
SNR = [10, 30, 50]
labels = ['fascicle 1', 'fascicle 2']

for i in range(3):
    to_plot1 = np.mean(reshape_ang_err2[0, i, :, :], 1)
    to_plot2 = np.mean(reshape_ang_err2[1, i, :, :], 1)
    
    ax1[i].plot(nu_min, to_plot1, color= 'purple', marker='x')
    ax1[i].plot(nu_min, to_plot2, color= 'olive', marker='x')

    ax1[i].set_title('SNR %s - 100' % (SNR[i]))
    ax1[i].yaxis.grid(True)
    ax1[i].set_xlabel('nu1')
    ax1[i].set_ylabel('Mean absolute error')
    ax1[i].set_ylim(0, 40)
  
fig1.legend(labels)
plt.savefig("graphs/AngErr2_SNRnu_test%s.pdf" %n_test, dpi=150) 
     
    