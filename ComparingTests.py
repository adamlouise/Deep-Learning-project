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
    filename = 'data_TEST1/DW_noisy_store_uniform_15000__lou_TEST1'
    y_data = pickle.load(open(filename, 'rb'))
    y_data = y_data/M0
    print('ok noise')
else:
    filename = 'data_TEST1/DW_image_store_uniform_15000__lou_TEST1'
    y_data = pickle.load(open(filename, 'rb'))
    print('ok no noise')

target_data = util.loadmat(os.path.join('data_TEST1',
                                            "training_datauniform_15000_samples_lou_TEST1"))

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

# small changes for nn
y_data = np.transpose(y_data)
y_data_n = torch.from_numpy(y_data)
y_data_n = y_data_n.float()
#y_data_n = torch.transpose(y_data_n, 0, 1) 

#%% Load NNLS data

new_gen=False
nouvel_enregist = False
via_pickle = True
filename1 = 'data_TEST1/dataNW2_w_store_TEST1'
filename2 = 'data_TEST1/dataNW2_targets_TEST1' 

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

PATH = "models_statedic/M2_version8_StateDict.pt"
net2.load_state_dict(torch.load(PATH))
net2.eval()

##-----Trees-----

# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from sklearn.multioutput import MultiOutputRegressor

# model_rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=40, max_depth=12, random_state=0))
# model_b = MultiOutputRegressor(XGBRegressor())

filename_rf = "models_statedic/M3_RandomForest_version8_1"
model_rf = pickle.load(open(filename_rf, 'rb'))

filename_b = "models_statedic/M3_GradientBoosting_version8_1"
model_b = pickle.load(open(filename_b, 'rb'))
    
#%% predictions

##-----NN-1-----

tic = time.time()
output1 = net1(y_data_n)
output1 = output1.detach().numpy()
toc = time.time()
predic_time1 = toc - tic

error1 = abs(output1 - target_params_y.T) #(15000, 6)
sample_error1 = np.mean(error1, 0)
error1_vec = np.mean(error1, 1)

print("-- NN1 -- \n", 
      "Mean error: ", np.mean(sample_error1), '\n'
      "Error prop: ", sample_error1, '\n',
      "prediction time: ", predic_time1, '\n')

##-----NN-2-----

tic = time.time()

w_test = np.zeros((num_sample, num_atoms, num_fasc))
w_test[:, :, 0] = w_store[:,0:num_atoms]
w_test[:, :, 1] = w_store[:,num_atoms:2*num_atoms]
w_test = torch.from_numpy(w_test).float()
print(w_test.shape)
    
output2 = net2(w_test[:,:,0], w_test[:,:,1])
output2 = output2.detach().numpy()

toc = time.time()
predic_time2 = toc - tic

target_params_w = target_params_w.detach().numpy()
error2 = abs(output2 - target_params_w) #(15000, 6)
sample_error2 = np.mean(error2, 0)
error2_vec = np.mean(error2, 1)

print("-- NN2 -- \n", 
      "Mean error: ", np.mean(sample_error2), '\n'
      "Error prop: ", sample_error2, '\n',
      "prediction time: ", predic_time2, '\n')

#%%-----Trees-----

## RF
tic = time.time()
output_rf = model_rf.predict(y_data) # y_data:(552, 15000)
toc = time.time()
predic_time_rf = toc - tic

error_rf = abs(output_rf - target_params_y.T) #(15000, 6)
sample_error_rf = np.mean(error_rf, 0)
error_rf_vec = np.mean(error_rf, 1)

print("-- RF -- \n", 
      "Mean error: ", np.mean(sample_error_rf), '\n'
      "Error prop: ", sample_error_rf, '\n',
      "prediction time: ", predic_time_rf, '\n')

## Boost
tic = time.time()
output_b = model_b.predict(y_data)
toc = time.time()
predic_time_b = toc - tic

error_b = abs(output_b - target_params_y.T) #(15000, 6)
sample_error_b = np.mean(error_b, 0)        #(6,)
error_b_vec = np.mean(error_b, 1)           #(15000,)

print("-- Boosting -- \n", 
      "Mean error: ", np.mean(sample_error_b), '\n'
      "Error prop: ", sample_error_b, '\n',
      "prediction time: ", predic_time_b, '\n')


#%% Graphs
SNR = [10, 30, 50]
nu_min = [0.5, 0.4, 0.3, 0.2, 0.1]

tab_error1 = np.zeros((3, 5, 1000))
tab_error2 = np.zeros((3, 5, 1000))
tab_error_rf = np.zeros((3, 5, 1000))
tab_error_b = np.zeros((3, 5, 1000))
for i in range(3):
    for j in range(5):
        elem = j*3 + i
        tab_error1[i, j, :] = error1_vec[elem*1000:(elem+1)*1000]
        tab_error2[i, j, :] = error2_vec[elem*1000:(elem+1)*1000]
        tab_error_rf[i, j, :] = error_rf_vec[elem*1000:(elem+1)*1000]
        tab_error_b[i, j, :] = error_b_vec[elem*1000:(elem+1)*1000]

prop_error1 = np.zeros((3, 5, 6))
prop_error2 = np.zeros((3, 5, 6))
prop_error_rf = np.zeros((3, 5, 6))
prop_error_b = np.zeros((3, 5, 6))
for i in range(3):
    for j in range(5):
        elem = j*3 + i
        prop_error1[i, j, :] = np.mean(error1[elem*1000:(elem+1)*1000, :], 0)
        prop_error2[i, j, :] = np.mean(error2[elem*1000:(elem+1)*1000, :],0)
        prop_error_rf[i, j, :] = np.mean(error_rf[elem*1000:(elem+1)*1000, :],0)
        prop_error_b[i, j, :] = np.mean(error_b[elem*1000:(elem+1)*1000, :],0)
        
prop_tot_error1 = np.zeros((3, 5, 1000, 6))
prop_tot_error2 = np.zeros((3, 5, 1000, 6))
prop_tot_error_rf = np.zeros((3, 5, 1000, 6))
prop_tot_error_b = np.zeros((3, 5, 1000, 6))
for i in range(3):
    for j in range(5):
        elem = j*3 + i
        prop_tot_error1[i, j, :, :] = error1[elem*1000:(elem+1)*1000, :]
        prop_tot_error2[i, j, :, :] = error2[elem*1000:(elem+1)*1000, :]
        prop_tot_error_rf[i, j, :, :] = error_rf[elem*1000:(elem+1)*1000, :]
        prop_tot_error_b[i, j, :, :] = error_b[elem*1000:(elem+1)*1000, :]

#%% graphe SNR

#colors = ['pink', 'lightblue', 'lightgreen', 'darkgreen']
colors = ['indianred', 'steelblue', 'lightgreen', 'green']
labels = ['NN1', 'NN2', 'RF', 'Boosting']

fig1, ax1 = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
fig1.suptitle('Error for different noise levels')

for i in range(3):
    bplot = ax1[i].boxplot([ np.matrix.flatten(tab_error1[i, :, :]), 
                           np.matrix.flatten(tab_error2[i, :, :]), 
                           np.matrix.flatten(tab_error_rf[i, :, :]),
                           np.matrix.flatten(tab_error_b[i, :, :])],
                           vert=True,  # vertical box alignment
                           widths = 0.3,
                           patch_artist=True,  # fill with color
                           labels=labels)  # will be used to label x-ticks
    ax1[i].set_title('SNR %s - 100' % (SNR[i]))
    
    for j in range(4):
        patch = bplot['boxes'][j]
        patch.set_facecolor(colors[j])

    ax1[i].yaxis.grid(True)
    #ax.set_xlabel('Three separate samples')
    ax1[i].set_ylabel('Mean absolute error')
    ax1[i].set_ylim(0, 1.3)
    
plt.savefig("graphs/Comp_SNR_test1.pdf", dpi=150) 

#%% graphe nus

colors = ['indianred', 'steelblue', 'limegreen', 'darkgreen']
fig2, ax2 = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
fig2.suptitle('Error dependent on nu for different noise levels')

for i in range(3):
    to_plot1 = np.mean(tab_error1[i, :, :], 1)
    to_plot2 = np.mean(tab_error2[i, :, :], 1)
    to_plot_rf = np.mean(tab_error_rf[i, :, :], 1)
    to_plot_b = np.mean(tab_error_b[i, :, :], 1)
    
    ax2[i].plot(nu_min, to_plot1, color= colors[0], marker='x')
    ax2[i].plot(nu_min, to_plot2, color= colors[1], marker='x')
    ax2[i].plot(nu_min, to_plot_rf, color= colors[2], marker='x')
    ax2[i].plot(nu_min, to_plot_b, color= colors[3], marker='x')

    ax2[i].set_title('SNR %s - 100' % (SNR[i]))


    ax2[i].yaxis.grid(True)
    ax2[i].set_xlabel('nu1')
    ax2[i].set_ylabel('Mean absolute error')
    ax2[i].set_ylim(0, 0.75)
    
fig2.legend(labels)
plt.savefig("graphs/Comp_Nus_test1.pdf", dpi=150) 

#%% Graphe prop vs SNR

colors = ['indianred', 'steelblue', 'limegreen', 'darkgreen']
fig3, ax3 = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
fig3.suptitle('Error of each property dependent on nu for different noise levels')
prop =['nu', 'rad', 'fin']
for i in range(3):
    for j in range(3):
    
        ax3[j,i].plot(nu_min, (prop_error1[i,:,j]+prop_error1[i,:,j+3])/2, color= colors[0], marker='x')
        ax3[j,i].plot(nu_min, (prop_error2[i,:,j]+prop_error2[i,:,j+3])/2, color= colors[1], marker='x')
        ax3[j,i].plot(nu_min, (prop_error_rf[i,:,j]+prop_error_rf[i,:,j+3])/2, color= colors[2], marker='x')
        ax3[j,i].plot(nu_min, (prop_error_b[i,:,j]+prop_error_b[i,:,j+3])/2, color= colors[3], marker='x')
        
        if i==0:
            ax3[j,i].set_ylabel('Mean absolute error \n %s' % (prop[j]))
        if j==0:
            ax3[j,i].set_title('SNR %s - 100' % (SNR[i]))
        if j==2:
            ax3[j,i].set_xlabel('nu1')
            
        ax3[j,i].yaxis.grid(True)
        ax3[j,i].set_ylim(0, 0.75)
    
fig3.legend(labels)
plt.savefig("graphs/Comp_propSNR_test1.pdf", dpi=150) 

#%% Graphe boxplot 

#colors = ['indianred', 'steelblue', 'limegreen', 'darkgreen']
colors = ['salmon', 'steelblue', 'lightgreen', 'green']
fig4, ax4 = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
fig4.suptitle('Error of each property dependent on nu for different noise levels')
prop =['nu', 'rad', 'fin']
for i in range(3): # i = SNR
    for l in range(3): # l = prop
    
        bplot = ax4[l,i].boxplot([ np.matrix.flatten((prop_tot_error1[i, :, :, l] + prop_tot_error1[i, :, :, l+3])/2), 
                           np.matrix.flatten((prop_tot_error2[i, :, :, l] + prop_tot_error2[i, :, :, l+3])/2), 
                           np.matrix.flatten((prop_tot_error_rf[i, :, :, l] + prop_tot_error_rf[i, :, :, l+3])/2),
                           np.matrix.flatten((prop_tot_error_b[i, :, :, l] + prop_tot_error_b[i, :, :, l+3])/2) ],
                           notch = True,
                           sym = "",
                           vert=True,  # vertical box alignment
                           widths = 0.25,
                           patch_artist=True,  # fill with color
                           labels=labels)  # will be used to label x-ticks
    
        for j in range(4): # number of methods
            patch = bplot['boxes'][j]
            patch.set_facecolor(colors[j])

        ax4[l,i].yaxis.grid(True)
        #ax.set_xlabel('Three separate samples')
        ax4[l,i].set_ylim(0, 1.4)
        
        if i==0:
            ax4[l,i].set_ylabel('Boxplot of error \n %s' % (prop[l]))
        if l==0:
            ax4[l,i].set_title('SNR %s - 100' % (SNR[i]))
            
        ax4[l,i].yaxis.grid(True)
    
plt.savefig("graphs/Comp_Boxplot_propSNR_test1.pdf", dpi=150) 