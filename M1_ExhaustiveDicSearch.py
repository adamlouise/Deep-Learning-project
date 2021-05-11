#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 10:10:12 2021

@author: louiseadam
"""

# Loading all the tools
import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch

# path to MF source files 
#path_to_utils = os.path.join('.', 'python_functions')
#path_to_utils = os.path.abspath(path_to_utils)
#if path_to_utils not in sys.path:
#    sys.path.insert(0, path_to_utils)
import mf_utils as util

import time

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

num_atoms = 782
num_fasc = 2

#%% Loading the dictionary and the DW-signals -- takes some time

def get_data(SNR_min=80, SNR_max=100, num_samples=1000):
    use_noise = True
    load_scaler = True
    
    # Load single-fascicle canonical dictionary and data gen function
    from get_data_test import gen_test_data, ld_singfasc_dic
        
    DW_image_store, DW_noisy_store, dic_compile, time_elapsed, sch_mat_b0, IDs, nus = gen_test_data(num_samples, use_noise, SNR_min, SNR_max)
    
    # verifier que ca fonctionne...
    target_params = np.zeros((6, num_samples))
    target_params[0,:] = nus[:,0]
    target_params[1,:] = ld_singfasc_dic['subinfo']['rad'][IDs[:,0]]
    target_params[2,:] = ld_singfasc_dic['subinfo']['fin'][IDs[:,0]]
    target_params[3,:] = nus[:,1]
    target_params[4,:] = ld_singfasc_dic['subinfo']['rad'][IDs[:,1]]
    target_params[5,:] = ld_singfasc_dic['subinfo']['fin'][IDs[:,1]]
    
    if load_scaler:
        scaler1 = pickle.load(open('scaler1_version1', 'rb'))
        target_params = scaler1.transform(target_params.T)
    else:       
        scaler1 = StandardScaler()
        target_params = scaler1.fit_transform(target_params.T)
    
    target_params = target_params.T
    
    return DW_noisy_store, target_params, dic_compile, sch_mat_b0, DW_image_store


#%% Test with M1 - exhaustive search

def method1(DW_store, dic_compile, num_samples):
    dicsizes = np.array([782, 782])
    
    tic = time.time()
    
    for i in range(num_samples):
        if i%10==0:
            print("sample:", i)
        y = DW_store[:,i]
        dictionary = dic_compile[i, :, :]
        w_nneg, ind_atoms_subdic, ind_atoms_totdic, min_obj, y_recons = util.solve_exhaustive_posweights(dictionary, y, dicsizes)
    
    toc = time.time()
    deltat = toc - tic
    
    print(deltat)
    #print(w_nneg)
    #print(ind_atoms_subdic)
    #print(ind_atoms_totdic)
    #print(min_obj)
    
    error1 = abs(y - y_recons)
    mean_error1 = np.mean(error1)
    
    return error1, deltat

#error1 = method1(DW_noisy_store)

#%% Test with M2 - NNLS + deep learning

from Classes.Net2_Class import create_Net2

def method2(DW_noisy_store, target_params, num_samples, dic_compile, sch_mat_b0):
    
    params2 = pickle.load(open('model2_29_params2', 'rb'))
    
    net2 = create_Net2(params2) 
    
    PATH = "M2_version1_StateDict.pt"
    net2.load_state_dict(torch.load(PATH))
    net2.eval()
        
    # Solve NNLS
    tic = time.time()
    
    w_test = np.zeros((num_samples, num_atoms, num_fasc))
    for i in range(num_samples):
        y = DW_noisy_store[:,i]
        dictionary = dic_compile[i, :, :]
        norm_y = np.max(y[sch_mat_b0[:, 3] == 0]) #Important???
        (w_nnls, PP, _) = util.nnls_underdetermined(dictionary, y/norm_y)
        
        w_test[i, :, 0] = w_nnls[0:num_atoms]
        w_test[i, :, 1] = w_nnls[num_atoms:2*num_atoms]
    
    w_test = torch.from_numpy(w_test).float()
    print(w_test.shape)
    # predict and time
    output2 = net2(w_test[:,:,0], w_test[:,:,1])
    output2 = output2.detach().numpy()
    
    toc = time.time()
    predic_time = toc - tic
    print("prediction time: ", predic_time)
    
    error2 = abs(output2.T - target_params)
    print(np.mean(error2))
    
    sample_error2 = np.mean(error2, 0)
    print(sample_error2.shape)
    
    return sample_error2, predic_time

#%% Test for trees

def method3(DW_noisy_store):
    filename_model3 = "M3_RandomForest"
    model3 = pickle.load(open(filename_model3, 'rb'))
    #trees = model3['model']
    
    # predict and time
    tic = time.time()
    output3 = model3(DW_noisy_store[:,0].T)
    toc = time.time()
    predic_time = toc - tic
    print("prediction time: ", predic_time)
    
    error3 = abs(output3 - target_params)
    
    return error3

#%% Test for Pure Deep Learning

from Classes.Net1_Class import create_Net1

def method4(DW_noisy_store, target_params, useNoise):
    
    params1 = pickle.load(open('model1_29_params1', 'rb'))
    
    net1 = create_Net1(params1)
    # load
    if useNoise == True:   
        PATH = "M1_version1_Noise_StateDict.pt"
        net1.load_state_dict(torch.load(PATH))
        net1.eval()
    else:
        PATH = "M1_version1_NoNoise_StateDict.pt"
        net1.load_state_dict(torch.load(PATH))
        net1.eval()
    
    M, num_samples = DW_noisy_store.shape
    DW_noisy_store = torch.from_numpy(DW_noisy_store)
    DW_noisy_store = DW_noisy_store.float()
    DW_noisy_store = torch.transpose(DW_noisy_store, 0, 1) 
    print(DW_noisy_store.shape)
    
    tic = time.time()
    output4 = net1(DW_noisy_store)
    output4 = output4.detach().numpy()
    toc = time.time()
    
    predic_time = toc - tic
    print("prediction time: ", predic_time)
    
    error4 = abs(output4.T - target_params)
    print(error4.shape)
    print(np.mean(error4))
    sample_error4 = np.mean(error4, 0)
    print("taille ",sample_error4.shape)
    
    for i in range(6):
        print(np.mean(error4[i,:]))
    
    return sample_error4
    

#%% Box plot

DW_noisy_store1, target_params1, dic_compile1, sch_mat_b0, DW_image_store1 = get_data(80, 100, 200)
DW_noisy_store2, target_params2, dic_compile2, sch_mat_b0, DW_image_store2 = get_data(40, 100, 200)

sample0_error4_Noise = method4(DW_image_store1, target_params1, True)
sample0_error4_NoNoise = method4(DW_image_store1, target_params1, False)

sample1_error4_Noise = method4(DW_noisy_store1, target_params1, True)
sample1_error4_NoNoise = method4(DW_noisy_store1, target_params1, False)

sample2_error4_Noise = method4(DW_noisy_store2, target_params2, True)
sample2_error4_NoNoise = method4(DW_noisy_store2, target_params2, False)

#%% - 1 - Boxplots 

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
labels = ['Training with noise', 'Training without noise']
fig.suptitle('Error for different noise levels')
bplot1 = ax1.boxplot([sample0_error4_Noise, sample0_error4_NoNoise],
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
ax1.set_title('No noise')

bplot2 = ax2.boxplot([sample1_error4_Noise, sample1_error4_NoNoise],
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
ax2.set_title('SNR 80-100')

bplot3 = ax3.boxplot([sample2_error4_Noise, sample2_error4_NoNoise],
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
ax3.set_title('SNR 40-100')

# fill with colors
# colors = ['pink', 'lightblue', 'lightgreen']
colors = ['pink', 'lightgreen']
for bplot in (bplot1, bplot2, bplot3):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

# adding horizontal grid lines
for ax in [ax1, ax2, ax3]:
    ax.yaxis.grid(True)
    #ax.set_xlabel('Three separate samples')
    ax.set_ylabel('Mean absolute error')
    ax.set_ylim(0, 1.3)

plt.show()

#%%

n_samples = 100
sample_error4_Noise = np.zeros(n_samples)
j=0
for i in [80]:
    DW_noisy_store, target_params, dic_compile, sch_mat_b0 = get_data(i, 100, n_samples)
    sample_error4_Noise = method2(DW_noisy_store, target_params, n_samples, dic_compile, sch_mat_b0)
    j = j+1

#%%
fig, ax1 = plt.subplots(nrows=1, ncols=1)

labels = ['SNR 80-100', 'SNR 50-100', 'SNR 20-100']
# rectangular box plot
bplot1 = ax1.boxplot([sample_error4_Noise[:,0], sample_error4_Noise[:,1], sample_error4_Noise[:,2]],
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
ax1.set_title('Error with method 4')

#fill with colors
colors = ['pink', 'lightblue', 'lightgreen']
for bplot in [bplot1]:
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

# adding horizontal grid lines
for ax in [ax1]:
    ax.yaxis.grid(True)
    #ax.set_xlabel('Three separate samples')
    ax.set_ylabel('Mean absolute error')
    ax.set_ylim(0, 1.5)

plt.show()





