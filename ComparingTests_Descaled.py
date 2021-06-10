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

n_test = '8_6_TEST2_descaled' # date for saving
num_sample = 15000 # size test set
n_sa = 1000
SNR = [10, 30, 50]
nu_min = [0.5, 0.4, 0.3, 0.2, 0.1]

num_params = 6
num_fasc = 2
M0 = 500
num_atoms = 782

n_SNR = len(SNR)
n_nu = len(nu_min)

load_scaler1 = True
load_scaler2 = True
scaled = False

#%% Load Dw_image data

use_noise = True
use_NoNoise = False
print("Noise", use_noise)

if use_noise:
    filename = 'data_TEST2/DW_noisy_store_uniform_15000__lou_TEST2'
    y_data = pickle.load(open(filename, 'rb'))
    y_data = y_data/M0
    # small changes for nn
    y_data = np.transpose(y_data)
    y_data_n = torch.from_numpy(y_data)
    y_data_n = y_data_n.float()   
    print('ok noise')
    
if use_NoNoise:   
    filename = 'data_TEST2/DW_image_store_uniform_15000__lou_TEST2'
    y_data2 = pickle.load(open(filename, 'rb'))    
    # small changes for nn
    y_data2 = np.transpose(y_data2)
    y_data2_n = torch.from_numpy(y_data2)
    y_data2_n = y_data2_n.float()
    print('ok no noise')
    
target_data = util.loadmat(os.path.join('data_TEST2',
                                            "training_datauniform_15000_samples_lou_TEST2"))

IDs = target_data['IDs'][0:num_sample, :]
nus = target_data['nus'][0:num_sample, :]
    
target_params_y = np.zeros((6, num_sample))

target_params_y[0,:] = nus[:,0]
target_params_y[1,:] = target_data['subinfo']['rad'][IDs[:,0]]
target_params_y[2,:] = target_data['subinfo']['fin'][IDs[:,0]]
target_params_y[3,:] = nus[:,1]
target_params_y[4,:] = target_data['subinfo']['rad'][IDs[:,1]]
target_params_y[5,:] = target_data['subinfo']['fin'][IDs[:,1]]


if load_scaler1:
    scaler1 = pickle.load(open('NN1_scaler1_version8', 'rb'))
else:
    scaler_y = StandardScaler()

baseline = np.mean(abs(target_params_y), 0)
mean_baseline_prop = np.mean(abs(target_params_y), 1)

#%% Load NNLS data

new_gen= False
nouvel_enregist = False
via_pickle = True
filename1 = 'data_TEST2/dataNW2_w_store_TEST2'
filename2 = 'data_TEST2/dataNW2_targets_TEST2' 

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


if load_scaler2:
    scaler_w = pickle.load(open('NN2_scaler2_version8', 'rb'))
    if scaled:
        target_params_w = scaler_w.transform(target_params_w)
else:
    scaler_w = StandardScaler()
    if scaled:
        target_params_w = scaler_w.fit_transform(target_params_w)

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

filename_rf = "models_statedic/M3_RandomForest_version8_1"
model_rf = pickle.load(open(filename_rf, 'rb'))

filename_b = "models_statedic/M3_GradientBoosting_version8_1"
#filename_b = "models_statedic/M3_GradientBoosting_version8_200000tsamples"
model_b = pickle.load(open(filename_b, 'rb'))

compare_NoNoise_Trees = True

if compare_NoNoise_Trees:
    filename_rf_NoNoise = "models_statedic/M3_RandomForest_version8_1NoNoise"
    model_rf_NoNoise = pickle.load(open(filename_rf_NoNoise, 'rb'))

    filename_b_NoNoise = "models_statedic/M3_GradientBoosting_version8_1NoNoise"
    model_b_NoNoise = pickle.load(open(filename_b_NoNoise, 'rb'))
    
    
#%% Load exhaustive search error

if scaled:
    filename0 = "error_M1_testdata_TEST2"
else:
    filename0 = 'error_M1_testdata_EstOrientations_%s' %n_test

error0 = pickle.load(open(filename0, 'rb'))
print(np.mean(error0,1))

scaled = False
if scaled:
    filename0_TrueOri = "error_M1_testdata_TEST2_TrueOrientations"
else:
    filename0_TrueOri = 'error_M1_testdata_TrueOrientations_%s' %n_test

error0_TrueOri = pickle.load(open(filename0_TrueOri, 'rb'))

print(np.mean(error0_TrueOri,1))
    
#%% predictions

##-----NN-1-----

def compute_error_NN1(y_data):
    tic = time.time()
    output1 = net1(y_data_n)
    output1 = output1.detach().numpy()
    
    if scaled==False: # alors je descale
        output1_desc = scaler1.inverse_transform(output1)
    
    toc = time.time()
    predic_time1 = toc - tic
    
    error1 = abs(output1_desc - target_params_y.T) #(15000, 6)
    sample_error1 = np.mean(error1, 0)
    error1_vec = np.mean(error1, 1)
    
    print("-- NN1 -- \n", 
          "Mean error: ", np.mean(sample_error1), '\n'
          "Error prop: ", sample_error1, '\n',
          "prediction time: ", predic_time1, '\n')
    
    
    
    return error1, error1_vec

##-----NN-2-----

def compute_error_NN2(w_store):
    tic = time.time()
    
    w_test = np.zeros((num_sample, num_atoms, num_fasc))
    w_test[:, :, 0] = w_store[:,0:num_atoms]
    w_test[:, :, 1] = w_store[:,num_atoms:2*num_atoms]
    w_test = torch.from_numpy(w_test).float()
    print(w_test.shape)
        
    output2 = net2(w_test[:,:,0], w_test[:,:,1])
    output2 = output2.detach().numpy()
    
    if scaled==False: # alors je descale
        output2_desc = scaler_w.inverse_transform(output2)
        
    #output2_desc = output2_desc.detach().numpy()
    
    toc = time.time()
    predic_time2 = toc - tic
    
    target_params_w2 = target_params_w.detach().numpy()
    error2 = abs(output2_desc - target_params_w2) #(15000, 6)
    sample_error2 = np.mean(error2, 0)
    error2_vec = np.mean(error2, 1)
    
    print("-- NN2 -- \n", 
          "Mean error: ", np.mean(sample_error2), '\n'
          "Error prop: ", sample_error2, '\n',
          "prediction time: ", predic_time2, '\n')
    
    return error2, error2_vec

#-----Trees-----

def compute_error_T(y_data, model, method=' '):
    
    tic = time.time()
    output = model.predict(y_data) # y_data:(552, 15000)
    toc = time.time()
    predic_time = toc - tic
    
    if scaled==False: # alors je descale
        output_desc = scaler1.inverse_transform(output)
    
    error = abs(output_desc - target_params_y.T) #(15000, 6)
    sample_error = np.mean(error, 0)
    error_vec = np.mean(error, 1)
    
    print("-- %s -- \n" %method, 
          "Mean error: ", np.mean(sample_error), '\n'
          "Error prop: ", sample_error, '\n',
          "prediction time: ", predic_time, '\n')
    
    return error, error_vec    

#%% Arrays for comparing 4 methods

def reshape(error, error_vec):
    tab_error = np.zeros((n_SNR, n_nu, n_sa))
    prop_error = np.zeros((n_SNR, n_nu, 6))
    prop_tot_error = np.zeros((n_SNR, n_nu, n_sa, 6))
    
    for i in range(n_SNR):
        for j in range(n_nu):
            elem = j*3 + i
            tab_error[i, j, :] = error_vec[elem*n_sa:(elem+1)*n_sa]
            prop_error[i, j, :] = np.mean(error[elem*n_sa:(elem+1)*n_sa, :], 0)
            prop_tot_error[i, j, :, :] = error[elem*n_sa:(elem+1)*n_sa, :]
            
    return tab_error, prop_error, prop_tot_error

error1, error1_vec = compute_error_NN1(y_data)
error2, error2_vec = compute_error_NN2(w_store)
error_rf, error_rf_vec = compute_error_T(y_data, model_rf, 'RF')    
error_b, error_b_vec = compute_error_T(y_data, model_b, 'Boosting')  

#baseline
#tab_b, prop_b, prop_tot_b = reshape(abs(target_params_y.T), np.mean(abs(target_params_y), 0))

#exhaustive search
tab_error0, prop_error0, prop_tot_error0 = reshape(error0.T, np.mean(error0, 0))
tab_error0_TrueOri, prop_error0_TrueOri, prop_tot_error0_TrueOri = reshape(error0_TrueOri.T, np.mean(error0_TrueOri, 0))

#methods
tab_error1, prop_error1, prop_tot_error1 = reshape(error1, error1_vec)
tab_error2, prop_error2, prop_tot_error2 = reshape(error2, error2_vec)
tab_error_rf, prop_error_rf, prop_tot_error_rf = reshape(error_rf, error_rf_vec)
tab_error_b, prop_error_b, prop_tot_error_b = reshape(error_b, error_b_vec)

if compare_NoNoise_Trees:
    error_rf_NoNoise, error_rf_vec_NoNoise = compute_error_T(y_data, model_rf_NoNoise, 'RF, _NoNoise')    
    error_b_NoNoise, error_b_vec_NoNoise = compute_error_T(y_data, model_b_NoNoise, 'Boosting, _NoNoise')  
    
    tab_error_rf_NoNoise, prop_error_rf_NoNoise, prop_tot_error_rf_NoNoise = reshape(error_rf_NoNoise, error_rf_vec_NoNoise)
    tab_error_b_NoNoise, prop_error_b_NoNoise, prop_tot_error_b_NoNoise = reshape(error_b_NoNoise, error_b_vec_NoNoise)



#%% All the graphs

# graphe SNR

#colors = ['pink', 'lightblue', 'lightgreen', 'darkgreen']
colors = ['black', 'indianred', 'steelblue', 'goldenrod']
labels = ['ES', 'NN1', 'NN2', 'Boosting']

fig1, ax1 = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
fig1.suptitle('Error for different noise levels')

for i in range(3):
    bplot = ax1[i].boxplot([ np.matrix.flatten(tab_error0[i, :, :]),
                            np.matrix.flatten(tab_error1[i, :, :]), 
                           np.matrix.flatten(tab_error2[i, :, :]),
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
    
plt.savefig("graphs/Comp_SNR_test%s.pdf" %n_test, dpi=150) 

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
plt.savefig("graphs/Comp_Nus_test%s.pdf" %n_test, dpi=150) 

#%% prop vs snr

fig3, ax3 = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
fig3.suptitle('Error of each property dependent on nu for different noise levels')
prop =['nu', 'rad', 'fin']
colors = ['grey','black', 'steelblue', 'goldenrod', 'indianred']
labels = ['ES*', 'ES', 'NNLS + DL', 'GBoosting', 'DL']
for i in range(3):
    for j in range(3): #prop
        
        ax3[j,i].plot(nu_min, (prop_error0_TrueOri[i,:,j]+prop_error0_TrueOri[i,:,j+3])/2, color= colors[0], marker='x')
        ax3[j,i].plot(nu_min, (prop_error0[i,:,j]+prop_error0[i,:,j+3])/2, color= colors[1], marker='x')
        ax3[j,i].plot(nu_min, (prop_error2[i,:,j]+prop_error2[i,:,j+3])/2, color= colors[2], marker='x')
        #ax3[j,i].plot(nu_min, (prop_error_rf[i,:,j]+prop_error_rf[i,:,j+3])/2, color= colors[2], marker='x')
        ax3[j,i].plot(nu_min, (prop_error_b[i,:,j]+prop_error_b[i,:,j+3])/2, color= colors[3], marker='x')
        ax3[j,i].plot(nu_min, (prop_error1[i,:,j]+prop_error1[i,:,j+3])/2, color= colors[4], marker='x')
        #ax3[j,i].plot(nu_min, (prop_b[i,:,j]+prop_b[i,:,j+3])/2, color= colors[4], marker='x')
        
        if i==0:
            ax3[j,i].set_ylabel('Mean absolute error \n \n %s' % (prop[j]))
        if j==0:
            ax3[j,i].set_title('SNR %s - 100' % (SNR[i]))
        if j==2:
            ax3[j,i].set_xlabel('nu1')
            
        ax3[j,i].yaxis.grid(True)
        if j==1:
            ax3[j,i].set_ylim(0, 0.0000015)
        else:    
            ax3[j,i].set_ylim(0, 0.25)
    
fig3.legend(labels)
plt.savefig("graphs/Comp_MAEprop_test%s.pdf" %n_test, dpi=150) 

#%% Comparing 2 fasc

fig31, ax31 = plt.subplots(nrows=3, ncols=2, figsize=(9, 12))
fig31.suptitle('Comparing the error of the 2 fascicles')
prop =['nu', 'rad', 'fin']
colors = ['grey', 'black', 'steelblue', 'goldenrod', 'indianred']
labels = ['ES*', 'ES', 'NNLS + DL', 'GBoosting', 'DL']
nu_min2 = [[0.5, 0.4, 0.3, 0.2, 0.1],
          [0.5, 0.6, 0.7, 0.8, 0.9]]
for i in range(2):
    for j in range(3): #prop
        ax31[j,i].plot(nu_min2[i], np.mean(prop_error0_TrueOri[:,:,i*3+j],0), color= colors[0], marker='x')
        ax31[j,i].plot(nu_min2[i], np.mean(prop_error0[:,:,i*3+j],0), color= colors[1], marker='x')
        ax31[j,i].plot(nu_min2[i], np.mean(prop_error2[:,:,i*3+j],0), color= colors[2], marker='x')
        #ax3[j,i].plot(nu_min, (prop_error_rf[i,:,j]+prop_error_rf[i,:,j+3])/2, color= colors[2], marker='x')
        ax31[j,i].plot(nu_min2[i], np.mean(prop_error_b[:,:,i*3+j],0), color= colors[3], marker='x')
        ax31[j,i].plot(nu_min2[i], np.mean(prop_error1[:,:,i*3+j],0), color= colors[4], marker='x')
        #ax3[j,i].plot(nu_min, (prop_b[i,:,j]+prop_b[i,:,j+3])/2, color= colors[4], marker='x')
        
        if i==0:
            ax31[j,i].set_ylabel('Mean absolute error \n \n %s' % (prop[j]))
        if j==0:
            ax31[j,i].set_title('fasc %s' % (i+1))
        if j==2:
            ax31[j,i].set_xlabel('nu %s' % (i+1))
            
        ax31[j,i].yaxis.grid(True)
        
        if j==1:
            ax31[j,i].set_ylim(0, 0.00000175)
        else:    
            ax31[j,i].set_ylim(0, 0.3)

    
fig31.legend(labels)
plt.savefig("graphs/Comp_fascProp_test%s.pdf" %n_test, dpi=150)


#%% Analyse de nu

fig3, ax3 = plt.subplots(nrows=2, ncols=3, figsize=(12, 9))
fig3.suptitle('Error of each property dependent on nu for different noise levels')
prop =['nu', 'rad', 'fin']
colors = ['black', 'steelblue', 'goldenrod', 'indianred']
labels = ['ES', 'NNLS', 'Boosting', 'DL']
fasc = ['fascicle 1', 'fascicle 2']
for i in range(3): #SNR
    for j in range(2): # prop
        
        ax3[j,i].plot(nu_min, prop_error0[i,:,j*3], color= colors[0], marker='x')
        ax3[j,i].plot(nu_min, prop_error2[i,:,j*3], color= colors[1], marker='x')
        #ax3[j,i].plot(nu_min, (prop_error_rf[i,:,j]+prop_error_rf[i,:,j+3])/2, color= colors[2], marker='x')
        ax3[j,i].plot(nu_min, prop_error_b[i,:,j*3], color= colors[2], marker='x')
        ax3[j,i].plot(nu_min, prop_error1[i,:,j*3], color= colors[3], marker='x')
        #ax3[j,i].plot(nu_min, (prop_b[i,:,j]+prop_b[i,:,j+3])/2, color= colors[4], marker='x')
        
        if i==0:
            ax3[j,i].set_ylabel('Mean absolute error \n \n %s' % fasc[j])
        if j==0:
            ax3[j,i].set_title('SNR %s - 100' % (SNR[i]))
        if j==2:
            ax3[j,i].set_xlabel('nu1')
            
        ax3[j,i].yaxis.grid(True)
        ax3[j,i].set_ylim(0, 1.65)
    
fig3.legend(labels)
plt.savefig("graphs/Comp_MAEnu2fasc_test%s.pdf" %n_test, dpi=150) 


#%% Graphe boxplot 

#colors = ['indianred', 'steelblue', 'limegreen', 'darkgreen']
colors = ['black','grey','skyblue', 'goldenrod', 'salmon']
labels = ['ES', 'ES*','NNLS', 'GBoost', 'DL']
fig4, ax4 = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
fig4.suptitle('Boxplot of error for each property and different noise levels',  y=0.945)
prop =['nu', 'rad', 'fin']
for i in range(3): # i = SNR
    for l in range(3): # l = prop
    
        bplot = ax4[l,i].boxplot([ np.matrix.flatten((prop_tot_error0[i, :, :, l] + prop_tot_error0[i, :, :, l+3])/2),
                                 np.matrix.flatten((prop_tot_error0_TrueOri[i, :, :, l] + prop_tot_error0_TrueOri[i, :, :, l+3])/2),
                                 np.matrix.flatten((prop_tot_error2[i, :, :, l] + prop_tot_error2[i, :, :, l+3])/2), 
                                 #np.matrix.flatten((prop_tot_error_rf[i, :, :, l] + prop_tot_error_rf[i, :, :, l+3])/2),
                                 np.matrix.flatten((prop_tot_error_b[i, :, :, l] + prop_tot_error_b[i, :, :, l+3])/2),
                                 np.matrix.flatten((prop_tot_error1[i, :, :, l] + prop_tot_error1[i, :, :, l+3])/2)],
                                 notch = True,
                                 sym = "",
                                 vert=True,  # vertical box alignment
                                 widths = 0.25,
                                 patch_artist=True,  # fill with color
                                 labels=labels,
                                 #positions = [-1, -0.65, -0.10, 0.45, 1]
                                 )  # will be used to label x-ticks
    
        for j in range(5): # number of methods
            patch = bplot['boxes'][j]
            patch.set_facecolor(colors[j])

        #ax4[l,i].set_xticklabels([]labels, rotation=15)
        ax4[l,i].yaxis.grid(True)
        #ax.set_xlabel('Three separate samples')
        ax4[l,i].set_ylim(0, 1.7)
        
        if i==0:
            ax4[l,i].set_ylabel('Boxplot of error \n %s' % (prop[l]))
        if l==0:
            ax4[l,i].set_title('SNR %s - 100' % (SNR[i]))
            
        ax4[l,i].yaxis.grid(True)
 
plt.savefig("graphs/Comp_Boxplot_propSNR_test%s.pdf" %n_test, dpi=150) 

#%% Graph for comparing training on noise vs no noise - boxplots

# error without noise
error_rf2, error_rf_vec2 = compute_error_T(y_data2, model_rf, 'RF')    
error_b2, error_b_vec2 = compute_error_T(y_data2, model_b, 'Boosting')  

error_rf2_NoNoise, error_rf_vec2_NoNoise = compute_error_T(y_data2, model_rf_NoNoise, 'RF _NoNoise')    
error_b2_NoNoise, error_b_vec2_NoNoise = compute_error_T(y_data2, model_b_NoNoise, 'Boosting _NoNoise')

SNR = [10, 30, 50]

#%%

fig5, ax5 = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
colors = ['lightpink', 'lightgreen', 'lightpink', 'lightgreen']
fig5.suptitle('Comparing mean error for traing on Noisy and Non Noisy data')
labels = ['Trained on Noise (50-100)', 'Trained without noise']
k=0 #graph

for i in [3, 2, 0]: # i = SNR
    
    if i==3:
        bplot = ax5[k].boxplot([ error_rf_vec2, error_rf_vec2_NoNoise,
                                error_b_vec2, error_b_vec2_NoNoise],
                           notch = True,
                           sym = "",
                           vert=True,  # vertical box alignment
                           widths = 0.22,
                           patch_artist=True,  # fill with color
                           positions = [-0.3, 0.3, 1.7, 2.3],
                           )  # will be used to label x-ticks
        ax5[k].set_title('No noise')
        ax5[k].set_ylabel('Boxplot of mean error' )
        
    else:
        bplot = ax5[k].boxplot([ np.matrix.flatten(tab_error_rf[i, :, :]),
                                np.matrix.flatten(tab_error_rf_NoNoise[i, :, :]), 
                                np.matrix.flatten(tab_error_b[i, :, :]),
                                np.matrix.flatten(tab_error_b_NoNoise[i, :, :]) ],
                               notch = True,
                               sym = "",
                               vert=True,  # vertical box alignment
                               widths = 0.22,
                               patch_artist=True,  # fill with color
                               positions = [-0.3, 0.3, 1.7, 2.3],
                               )  # will be used to label x-ticks
        ax5[k].set_title('SNR %s - 100' % (SNR[i]))

    #ax5[k].set_xticks(ticks)
    #ax5[k].set_xticklabels(lab)
    for j in range(4): # number of boxes
        patch = bplot['boxes'][j]
        patch.set_facecolor(colors[j])

    ax5[k].yaxis.grid(True)
    ax5[k].set_ylim(0, 1.2)

    k= k+1

plt.setp(ax5, xticks=[0, 2], xticklabels=['RF', 'GBoost'])
fig5.legend([bplot["boxes"][0], bplot["boxes"][1]], labels, loc='upper right')
plt.savefig("graphs/M3_Boxplot_NoiseTraining_test%s.pdf" %n_test, dpi=150) 


#%% ES: comparer les exhaustive search 3x3

colors = ['black', 'grey']
fig6, ax6 = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
fig6.suptitle('Comparing exhaustive search with perfect and estimated orientations', y=0.945)
prop =['nu', 'rad', 'fin']
labels = ['Estimated orientations', 'True orientations']
for i in range(3):
    for j in range(3):
        
        ax6[j,i].plot(nu_min, (prop_error0[i,:,j]+prop_error0[i,:,j+3])/2, color= colors[0], marker='x')
        ax6[j,i].plot(nu_min, (prop_error0_TrueOri[i,:,j]+prop_error0_TrueOri[i,:,j+3])/2, color= colors[1], marker='x')
        
        
        if i==0:
            ax6[j,i].set_ylabel('Mean absolute error \n \n %s' % (prop[j]))
        if j==0:
            ax6[j,i].set_title('SNR %s - 100' % (SNR[i]))
        if j==2:
            ax6[j,i].set_xlabel('nu1')
            
        ax6[j,i].yaxis.grid(True)
        ax6[j,i].set_ylim(0, 1.65)

    
fig6.legend(labels, loc=(0.8, 0.93))
#plt.title('Comparing exhaustive search with perfect and estimated orientations')
plt.savefig("graphs/ES_InfluenceOri_test%s.pdf" %n_test, dpi=150) 

#%% ES boxplots

fig7, ax7 = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
colors = ['black', 'grey']
fig7.suptitle('Influence of Orientation estimation on Exhaustive search', y=0.97)
lab = ['Estimated orientation', 'True orientation']

for i in range(3): # i = SNR
    
    bplot = ax7[i].boxplot([ np.matrix.flatten(tab_error0[i, :, :]),
                            np.matrix.flatten(tab_error0_TrueOri[i, :, :])],
                       notch = True,
                       sym = "",
                       vert=True,  # vertical box alignment
                       widths = 0.2,
                       patch_artist=True,  # fill with color
                       positions = [-0.5, 0.5],
                       labels=lab
                       )  # will be used to label x-ticks
    ax7[i].set_ylabel('Boxplot of mean error' )
    
    ax7[i].set_title('SNR %s - 100' % (SNR[i]))

    #ax5[k].set_xticks(ticks)
    #ax5[k].set_xticklabels(lab)
    for j in range(2): # number of boxes
        patch = bplot['boxes'][j]
        patch.set_facecolor(colors[j])

    ax7[i].yaxis.grid(True)
    ax7[i].set_ylim(0, 1.2)


#plt.setp(ax5, xticks=[0, 2], xticklabels=labels)
#fig7.legend([bplot["boxes"][0], bplot["boxes"][1]], labels, loc='upper right')
plt.savefig("graphs/ES_Boxplot_TrueOri_test%s.pdf" %n_test, dpi=150) 

#%% ES Comparin boxplots

fig8, ax8 = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
fig8.suptitle('Influence of Orientation estimation on Exhaustive search', y=0.97)

lab = ['Estimated orientation', 'True orientation']
prop =['nu', 'rad', 'fin']
colors = ['black', 'grey']

for i in range(3): # i = SNR
    for l in range(3): # l = prop
    
        bplot = ax8[l,i].boxplot([ np.matrix.flatten((prop_tot_error0[i, :, :, l] + prop_tot_error0[i, :, :, l+3])/2),
                                  np.matrix.flatten((prop_tot_error0_TrueOri[i, :, :, l] + prop_tot_error0_TrueOri[i, :, :, l+3])/2)],
                                 notch = True,
                                 sym = "",
                                 vert=True,  # vertical box alignment
                                 widths = 0.2,
                                 patch_artist=True,  # fill with color
                                 labels=lab)  # will be used to label x-ticks
    
        for j in range(2): # number of methods
            patch = bplot['boxes'][j]
            patch.set_facecolor(colors[j])

        ax8[l,i].yaxis.grid(True)
        #ax.set_xlabel('Three separate samples')
        ax8[l,i].set_ylim(0, 1.6)
        
        if i==0:
            ax8[l,i].set_ylabel('Boxplot of error \n %s' % (prop[l]))
        if l==0:
            ax8[l,i].set_title('SNR %s - 100' % (SNR[i]))
            
        ax8[l,i].yaxis.grid(True)
    
plt.savefig("graphs/ES_Boxplot_propSNR_test%s.pdf" %n_test, dpi=150) 

#%% Comparing ES boxplots

fig9, ax9 = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
colors = ['black', 'grey']
fig9.suptitle('Influence of Orientation estimation on Exhaustive search for each property', y=0.97)
lab = ['Estimated orientation', 'True orientation']
prop =['nu', 'rad', 'fin']

for i in range(3): # i = SNR
    
    bplot = ax9[i].boxplot([ np.matrix.flatten((prop_tot_error0[:, :, :, i] + prop_tot_error0[:, :, :, i+3])/2),
                            np.matrix.flatten((prop_tot_error0_TrueOri[:, :, :, i] + prop_tot_error0_TrueOri[:, :, :, i+3])/2)],
                       notch = True,
                       sym = "",
                       vert=True,  # vertical box alignment
                       widths = 0.2,
                       patch_artist=True,  # fill with color
                       positions = [-0.5, 0.5],
                       labels=lab
                       )  # will be used to label x-ticks
    ax9[i].set_ylabel('Boxplot of mean error' )
    
    ax9[i].set_title('property %s' % (prop[i]))

    for j in range(2): # number of boxes
        patch = bplot['boxes'][j]
        patch.set_facecolor(colors[j])

    ax9[i].yaxis.grid(True)
    ax9[i].set_ylim(0, 2)


#plt.setp(ax5, xticks=[0, 2], xticklabels=labels)
#fig7.legend([bplot["boxes"][0], bplot["boxes"][1]], labels, loc='upper right')
plt.savefig("graphs/ES_Boxplot_TrueOri_BoxplotsforProp_test%s.pdf" %n_test, dpi=150) 

#%% ES: comparer les exhaustive search 3x3

colors = ['purple', 'olive', 'black', 'grey']
fig6, ax6 = plt.subplots(nrows=4, ncols=3, figsize=(12, 15))
fig6.suptitle('Comparing exhaustive search with perfect and estimated orientations', y=0.945)
prop =['nu', 'rad', 'fin']
labels = ['AngErr fasc1', 'AngErr fasc2', 'Error with esti orient', 'Error with true orient']
for i in range(3): #SNR

    for k in range(4): #prop
    
        if k==0: 
            ax6[k,i].plot(nu_min, np.mean(reshape_ang_err2[0, i, :, :], 1), 
                          color= colors[0], marker='x', label = labels[0])
            ax6[k,i].plot(nu_min, np.mean(reshape_ang_err2[1, i, :, :], 1), 
                          color= colors[1], marker='x', label = labels[1])
            
            ax6[k,i].set_title('SNR %s - 100' % (SNR[i]))
            ax6[k,i].yaxis.grid(True)
            #ax6[k,i].set_xlabel('nu1')
            if i==0:
                ax6[k,i].set_ylabel('Angular error \n ')
            ax6[k,i].set_ylim(0, 40)
        
        else:
            j=k-1
            ax6[k,i].plot(nu_min, (prop_error0[i,:,j]+prop_error0[i,:,j+3])/2, 
                          color= colors[2], marker='x', label = labels[2])
            ax6[k,i].plot(nu_min, (prop_error0_TrueOri[i,:,j]+prop_error0_TrueOri[i,:,j+3])/2, 
                          color= colors[3], marker='x', label = labels[3])
            
            
            if i==0:
                ax6[k,i].set_ylabel('Mean absolute error \n \n %s' % (prop[j]))
            if j==2:
                ax6[k,i].set_xlabel('nu1')
                
            ax6[k,i].yaxis.grid(True)
            ax6[k,i].set_ylim(0, 1.65)

lines = []
labels = []
for ax in [ax6[0,0], ax6[1,0]]:
    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)

fig6.legend(lines, labels, loc=(0.8, 0.92))
#plt.title('Comparing exhaustive search with perfect and estimated orientations')
plt.savefig("graphs/ES_InfluenceOri_and_AngErr_test%s.pdf" %n_test, dpi=150) 