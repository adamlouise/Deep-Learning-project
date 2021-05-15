#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 16:47:44 2021

@author: louiseadam
"""

import mf_utils as util
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
import time

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

num_sample = 15000
num_params = 6
num_fasc = 2
M0 = 500
num_atoms = 782

#%% Load Dw_image data

use_noise = True
print("Noise", use_noise)

if use_noise:
    filename = 'data_TEST1/DW_noisy_store_uniform_600000__lou_TEST1'
    y_data = pickle.load(open(filename, 'rb'))
    y_data = y_data/M0
    print('ok noise')
else:
    filename = 'data_TEST1/DW_image_store_uniform_600000__lou_TEST1'
    y_data = pickle.load(open(filename, 'rb'))
    print('ok no noise')

data_dir = 'synthetic_data'

target_data = util.loadmat(os.path.join('synthetic_data',
                                            "training_datauniform_600000_samples_lou_version8"))

IDs = target_data['IDs'][0:num_sample, :]
nus = target_data['nus'][0:num_sample, :]
    
target_params_y = np.zeros((6, num_sample))

target_params_y[0,:] = nus[:,0]
target_params_y[1,:] = target_data['subinfo']['rad'][IDs[:,0]]
target_params_y[2,:] = target_data['subinfo']['fin'][IDs[:,0]]
target_params_y[3,:] = nus[:,1]
target_params_y[4,:] = target_data['subinfo']['rad'][IDs[:,1]]
target_params_y[5,:] = target_data['subinfo']['fin'][IDs[:,1]]

scaler_y = StandardScaler()
target_params_y = scaler_y.fit_transform(target_params_y.T)
target_params_y = target_params_y.T

#%% Load NNLS data

new_gen=True
nouvel_enregist = True
via_pickle = False
filename1 = 'data_TEST1/dataNW2_w_store_TEST1'
filename2 = 'data_TEST1/dataNW2_targets_version8_TEST1' 

if new_gen:   
    print("on load avec gen_batch_data")    
    from getDataW import gen_batch_data
    w_store, target_params_w = gen_batch_data(0, num_sample, 'train')
    print(w_store.shape, target_params_w.shape)
    
    if nouvel_enregist:
        print('et on enregistre :-) ')
        with open(filename1, 'wb') as f:
                pickle.dump(w_store, f)
                f.close()
        with open(filename2, 'wb') as f:
                pickle.dump(target_params_w, f)
                f.close()

if via_pickle:   
    print("on load via les fichiers pickle :-) ")     
    w_store = pickle.load(open(filename1, 'rb'))
    target_params_w = pickle.load(open(filename2, 'rb'))

scaler_w = StandardScaler()
target_params_w = scaler_w.fit_transform(target_params_w)
target_params_w = torch.from_numpy(target_params_w)    

#%% Load models

##-----NN-1-----

from Classes.Net1_Class import create_Net1

params1 = pickle.load(open('params/M1_params_16', 'rb'))
net1 = create_Net1(params1)

use_noise_NN1 = True

if use_noise_NN1 == True:   
    PATH = "models_statedic/M1_Noise_StateDict_version8.pt"
    net1.load_state_dict(torch.load(PATH))
    net1.eval()
else:
    PATH = ""
    net1.load_state_dict(torch.load(PATH))
    net1.eval()


##-----NN-2-----

from Classes.Net2_Class import create_Net2

params2 = pickle.load(open('params/M2_params_16', 'rb'))
net2 = create_Net2(params2)

PATH = "models_statedic/M2_Noise_StateDict_version8.pt"
net2.load_state_dict(torch.load(PATH))
net2.eval()

##-----Trees-----

filename_rf = "models_statedic/M3_RandomForest_version8_1"
model_rf = pickle.load(open(filename_rf, 'rb'))

filename_b = "models_statedic/M3_GradientBoosting_version8_1"
model_b = pickle.load(open(filename_b, 'rb'))
    
#%% predictions

##-----NN-1-----

y_data = torch.from_numpy(y_data)
y_data = y_data.float()
y_data = torch.transpose(y_data, 0, 1) 

tic = time.time()
output1 = net1(y_data)
output1 = output1.detach().numpy()
toc = time.time()
predic_time1 = toc - tic

error1 = abs(output1.T - target_params_y)
sample_error1 = np.mean(error1, 0)

print("prediction time: ", predic_time1)


##-----NN-2-----

tic = time.time()
output2 = net2(w_store )
output2 = output2.detach().numpy()
toc = time.time()
predic_time2 = toc - tic

error2 = abs(output2.T - target_params_w)
sample_error2 = np.mean(error2, 0)

print("prediction time: ", predic_time2)

##-----Trees-----

#%% Graphs