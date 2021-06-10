#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:23:10 2020

@author: louiseadam

NW1 
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from torch.nn.parameter import Parameter

path_to_utils = os.path.join('.', 'python_functions')
path_to_utils = os.path.abspath(path_to_utils)
if path_to_utils not in sys.path:
    sys.path.insert(0, path_to_utils)

import mf_utils as util
import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import time

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# %% Basic parameters

use_noise = True
num_sample = 600000
num_params = 6
num_fasc = 2
M0 = 500
num_atoms = 782
M = 552

# Saving
save_params = False
save_scaler = False
save_res = False
save_net = False
TESTNUM = '9_6'


num_div = num_sample/6
num_train = int(4*num_div)
num_test = int(num_train + num_div)
num_valid = int(num_test + num_div)

# Hyperparameter dictionary 
params1 = {
    #Training parameters
    "num_samples": num_sample,
     "batch_size": 5000,  #2500
     "num_epochs": 35,
     
     #NW2
     "num_h1": 300,
     "num_h2": 800,
     "num_h3": 1600,
     "num_h4": 800,
     "num_h5": 100,
     
     #other
     "learning_rate": 0.0005, #0.0005
     #"learning_rate": hp.choice("learningrate", [0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002]),
     "dropout": 0.05
     #"dropout": hp.uniform("dropout", 0, 0.4)
}

if save_params:
    filename = 'params/M1_params_%s' %TESTNUM 
    with open(filename, 'wb') as f:
              pickle.dump(params1, f)
              f.close()

#%% Load and reshape DW-MRI signals

if use_noise:
    filename = 'synthetic_data/DW_noisy_store_uniform_600000__lou_version8'
    y_data = pickle.load(open(filename, 'rb'))
    y_data = y_data/M0
    print('ok noise')
else:
    filename = 'synthetic_data/DW_image_store_uniform_600000__lou_version8'
    y_data = pickle.load(open(filename, 'rb'))
    print('ok no noise')

# divide data in train, test and validation
x_train = y_data[:, 0:num_train]
x_test = y_data[:, num_train : num_test ]
x_valid = y_data[:, num_test : num_valid ]

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
x_valid = torch.from_numpy(x_valid)

# modifications for neural network
x_train = x_train.float()
x_train = torch.transpose(x_train, 0, 1) 
x_test = x_test.float()
x_test = torch.transpose(x_test, 0, 1) 
x_valid = x_valid.float()
x_valid = torch.transpose(x_valid, 0, 1) 


# %% Loading and scaling target data

print("--- Taking microstructural properties of fascicles ---")

target_data = util.loadmat(os.path.join('synthetic_data',
                                            "training_datauniform_600000_samples_lou_version8"))

# Substrate (=fingerprint) properties
IDs = target_data['IDs'][0:num_sample, :]
nus = target_data['nus'][0:num_sample, :]
target_params = np.zeros((6, num_sample))

target_params[0,:] = nus[:,0]
target_params[1,:] = target_data['subinfo']['rad'][IDs[:,0]]
target_params[2,:] = target_data['subinfo']['fin'][IDs[:,0]]
target_params[3,:] = nus[:,1]
target_params[4,:] = target_data['subinfo']['rad'][IDs[:,1]]
target_params[5,:] = target_data['subinfo']['fin'][IDs[:,1]]

# Scaling: Standardisation of microstructural properties
scaler1 = StandardScaler()
target_params = scaler1.fit_transform(target_params.T)
target_params = target_params.T

if save_scaler:
    filename = "NN1_scaler1_version8_%s" %TESTNUM
    with open(filename, 'wb') as f:
              pickle.dump(scaler1, f)
              f.close()

# Dividing in train test and valid
target_train = target_params[:, 0:num_train]
target_test = target_params[:, num_train : num_test ]
target_valid = target_params[:, num_test : num_valid ]

# modifications for neural network
target_train = torch.from_numpy(target_train).float()
target_train = torch.transpose(target_train, 0, 1) 
target_test = torch.from_numpy(target_test).float()
target_test = torch.transpose(target_test, 0, 1) 
target_valid = torch.from_numpy(target_valid).float()
target_valid = torch.transpose(target_valid, 0, 1) 


# %% Building the network

class Net1(nn.Module):

    def __init__(self, num_in, num_h1, num_h2, num_h3, num_h4, num_h5, num_out, drop_prob):
        super(Net1, self).__init__()  
        # input layer
        self.W_1 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h1, num_in)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_h1), 0))
        self.l1_bn = nn.BatchNorm1d(num_h1)
        
        # hidden layers
        self.W_2 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h2, num_h1)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_h2), 0))
        self.l2_bn = nn.BatchNorm1d(num_h2)
        
        self.W_3 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h3, num_h2)))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_h3), 0))
        
        self.W_4 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h4, num_h3)))
        self.b_4 = Parameter(init.constant_(torch.Tensor(num_h4), 0))
        
        self.W_5 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h5, num_h4)))
        self.b_5 = Parameter(init.constant_(torch.Tensor(num_h5), 0))
        
        self.W_6 = Parameter(init.kaiming_uniform_(torch.Tensor(num_out, num_h5)))
        self.b_6 = Parameter(init.constant_(torch.Tensor(num_out), 0))
        
        #self.W_3_bn = nn.BatchNorm2d(num_out)
        # define activation function in constructor
        self.activation = torch.nn.ReLU()
        
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):

        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        x = self.dropout(x)

        x = F.linear(x, self.W_2, self.b_2)
        #x = self.l1_bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = F.linear(x, self.W_3, self.b_3)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = F.linear(x, self.W_4, self.b_4)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = F.linear(x, self.W_5, self.b_5)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = F.linear(x, self.W_6, self.b_6)

        return x

# %% Building training loop

def train_network1(params1: dict):

    num_in = 552
    num_out = num_params
    num_h1 = params1["num_h1"]
    num_h2 = params1["num_h2"]
    num_h3 = params1["num_h3"] 
    num_h4 = params1["num_h4"]
    num_h5 = params1["num_h5"]
    drop_prob = params1["dropout"]
    
    net1 = Net1(num_in, num_h1, num_h2, num_h3, num_h4, num_h5, num_out, drop_prob)
    
    # Optimizer and Criterion
    optimizer = optim.Adam(net1.parameters(), lr=params1["learning_rate"], weight_decay=0.0000001)
    lossf = nn.MSELoss()

    print('----------------------- Training --------------------------')
    
    # setting hyperparameters and getting epoch sizes
    batch_size = params1["batch_size"] 
    num_epochs = params1["num_epochs"] 
    num_samples_train = x_train.shape[0]
    num_batches_train = num_samples_train // batch_size 
    num_samples_valid = x_valid.shape[0]
    num_batches_valid = num_samples_valid // batch_size
    
    # setting up lists for handling loss/accuracy
    train_acc = np.zeros((num_epochs, num_params))
    valid_acc = np.zeros((num_epochs, num_params))
    meanTrainError, meanValError  = [], []
    cur_loss = 0
    losses = []
    
    start_time = time.time()

    get_slice = lambda i, size: range(i * size, (i + 1) * size)
    
    for epoch in range(num_epochs): # Forward -> Backprob -> Update params
        
        ## Train
        cur_loss = 0
        net1.train()
        for i in range(num_batches_train):
            
            optimizer.zero_grad()
            slce = get_slice(i, batch_size)
            output = net1(x_train[slce])
            
            # compute gradients given loss
            target_batch = target_train[slce]
            batch_loss = lossf(output, target_batch)
            batch_loss.backward()
            optimizer.step()
            
            cur_loss += batch_loss   
        losses.append(cur_loss / batch_size)
    
        net1.eval()
        
        ### Evaluate training
        train_preds = [[], [], [], [], [], []]
        train_targs = [[], [], [], [], [], []]
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            preds = net1(x_train[slce, :])
            
            for j in range(num_params):
                train_targs[j] += list(target_train[slce, j].numpy())
                train_preds[j] += list(preds.data[:,j].numpy())
            
        ### Evaluate validation
        val_preds = [[], [], [], [], [], []]
        val_targs = [[], [], [], [], [], []]
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            preds = net1(x_valid[slce, :])
            
            for j in range(num_params):
                val_targs[j] += list(target_valid[slce, j].numpy())
                val_preds[j] += list(preds.data[:,j].numpy())
                
        # Save evaluation and training
        train_acc_cur = np.zeros(num_params)
        valid_acc_cur = np.zeros(num_params)
        for j in range(num_params):
            train_acc_cur[j] = mean_absolute_error(train_targs[j], train_preds[j])
            valid_acc_cur[j] = mean_absolute_error(val_targs[j], val_preds[j])
            train_acc[epoch, j] = train_acc_cur[j]
            valid_acc[epoch, j] = valid_acc_cur[j]
        
        meanTrainError.append(np.mean(train_acc[epoch,:]))
        meanValError.append(np.mean(valid_acc[epoch, :]))
        
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f, " %(
            epoch+1, losses[-1], meanTrainError[-1], meanValError[-1]))
        
        t = time.time() - start_time
        print("time", t)
        
    to_min = sum(valid_acc_cur)
      
    return {"loss": to_min, 
            "model": net1, 
            "params": params1, 
            "status": STATUS_OK,
            "train_acc": train_acc,
            "valid_acc": valid_acc,
            "meanTrainError": meanTrainError,
            "meanValError": meanValError
            }

#%% Train & save models and results

trial = train_network1(params1)  
   
if save_res:     
    # Save trial for results
    filename = "results/M1_trial_version8_%s" %TESTNUM
    with open(filename, 'wb') as f:
        pickle.dump(trial, f)
        f.close()

if save_net:
    # Save net with state dictionary
    PATH = "models_statedic/M1_Noise_StateDict_version8_%s.pt" %TESTNUM
    net = trial['model']
    torch.save(net.state_dict(), PATH)
    
        
#%% Graphs for Learning

train_acc = trial['train_acc']
valid_acc = trial['valid_acc']
epoch = np.arange(params1['num_epochs'])

meanTrainError = trial['meanTrainError']
meanValError = trial['meanValError']

labels = ['nu', 'radius', 'fin']

## - 1 - Graph for Learning curve of 6 properties

fig, axs = plt.subplots(2, 3, sharey='row', sharex = 'col', figsize=(11,7))
fig.suptitle('Learning curve for each property and each fascicle')
for i in range(2):
    for j in range(3):    
        axs[i,j].plot(epoch, train_acc[:, j], 'r', epoch, valid_acc[:, j], 'b')
        axs[i,j].axis([0, len(epoch), 0, 0.6])
        if j==0:
            axs[i,j].set_ylabel('Absolute Error')
        if i==1:
            axs[i,j].set_xlabel('Epochs')
        axs[i,j].set_title(labels[j] + ' for fascicle '+ str(i+1))
        #axs[i,j].grid()
        axs[i,j].grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.4)
        axs[i,j].minorticks_on()
        axs[i,j].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        #axs[i,j].grid()
        #axs[i,j].legend(['Train error','Validation error'])
fig.legend(['Train error','Validation error'])     

plt.savefig("graphs/NN1_LC_6properties_%s.pdf" %TESTNUM, dpi=150)  

## - 2 - Graoh for learning curve of Mean Error

fig3, axs3 = plt.subplots(1, 1)

fig3.suptitle('Learning curve with 400 000 training samples')
axs3.plot(epoch, meanTrainError, 'r', epoch, meanValError, 'b')
axs3.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.4)
axs3.minorticks_on()
axs3.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#axs3.grid()
fig3.legend(['Mean Train error','Mean Validation error'])
axs3.set_xlabel('Epochs'), 
axs3.set_ylabel('Mean Scaled Error')
axs3.axis([0, len(epoch)-5, 0, 0.6])

plt.savefig("graphs/NN1_LC_MeanError_%s.pdf" %TESTNUM, dpi=150) 

## - 3 - Graph for Learning curve of 3 properties (mean over fascicles)

fig2, axs2 = plt.subplots(1, 3, sharey='row', figsize=(15,4))
fig2.suptitle('Learning curve for each property - mean over fascicles')
for j in range(3):    
    axs2[j].plot(epoch, (train_acc[:, j]+train_acc[:, j+3])/2, 'r', epoch, (valid_acc[:, j]+valid_acc[:, j+3])/2, 'b')
    axs2[j].axis([0, len(epoch), 0, 0.6])
    if j==0:
        axs2[j].set_ylabel('Absolute Error')
    axs2[j].set_xlabel('Epochs')
    axs2[j].set_title(labels[j])
    axs2[j].grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.4)
    axs2[j].minorticks_on()
    axs2[j].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    #axs2[j].grid()

fig2.legend(['Train error','Validation error'])  

plt.savefig("graphs/NN1_LC_3prop_%s.pdf" %TESTNUM, dpi=150) 


#%% Predictions
      
# predict
net = trial['model']
output = net(x_test)
output = output.detach().numpy()

mean_err_scaled = np.zeros(6)
for i in range(6):
    mean_err_scaled[i] = mean_absolute_error(output[:,i], target_test[:,i])

properties = ['nu 1', 'rad 1', 'fin 1', 'nu 2', 'rad 2', 'fin 2']
plt.figure()
plt.bar(properties, mean_err_scaled, width=0.5)
plt.savefig("graphs/NN1_bars_%s.pdf" %TESTNUM, dpi=150)  

# 95% interval
from scipy import stats

output = scaler1.inverse_transform(output)
target_scaled = scaler1.inverse_transform(target_test)

error = output - target_scaled
conf_int = np.zeros(num_params)

for j in range(num_params):
    data = error[:,j]    
    mean = np.mean(data)
    sigma = np.std(data)   
    confint = stats.norm.interval(0.95, loc=mean, scale=sigma)   
    print(confint)
    print((-confint[0]+confint[1])/2)


#%% ##### HYPEROPTI ######
# recompile hyperparameter dictionary!!

# 1 # Opti dropout

#dropout = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
dropout = []
n = len(dropout)

error_d = np.zeros((2, n))
for i in range(n):
    params1['dropout'] = dropout[i]
    tic = time.time()
    trial = train_network1(params1)
    #net_tot = trial['model']
    toc = time.time()
    train_time = toc - tic
    print("training time: ", train_time)
    
    error_d[0, i] = trial['meanValError'][-1]
    error_d[1, i] = trial['meanTrainError'][-1]
    
    print("Okayy -- ", error_d[0,i], error_d[1,i])

print("Dropout fini :-)", error_d)
params1['dropout'] = 0.05

# 2 # Opti learning rate

#lr = [0.0005, 0.0015, 0.0025, 0.005, 0.01]
lr = [0.0001, 0.00002, 0.00001]
n = len(lr)

error_lr = np.zeros((2, n))
for i in range(n):
    params1['learning_rate'] = lr[i]
    tic = time.time()
    trial = train_network1(params1)
    #net_tot = trial['model']
    toc = time.time()
    train_time = toc - tic
    print("training time: ", train_time)
    
    error_lr[0, i] = trial['meanValError'][-1]
    error_lr[1, i] = trial['meanTrainError'][-1]
    
    print("Okayy -- ", error_lr[0,i], error_lr[1,i])

print("lr fini :-)", error_lr)
params1['learning_rate'] = 0.001

# 3 # Opti batch

#batch = [500, 1000, 2000, 5000, 10000]
batch = []
n = len(batch)

error_batch = np.zeros((2, n))
for i in range(n):
    params1['batch_size'] = batch[i]
    tic = time.time()
    trial = train_network1(params1)
    #net_tot = trial['model']
    toc = time.time()
    train_time = toc - tic
    print("training time: ", train_time)
    
    error_batch[0, i] = trial['meanValError'][-1]
    error_batch[1, i] = trial['meanTrainError'][-1]
    
    print("Okayy -- ", error_batch[0,i], error_batch[1,i])

print("batch fini :-)", error_batch)

#%% Graph hyperopti

dropout = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
error_d = [[0.25276598, 0.23289584, 0.24382359, 0.25945747, 0.27979863,
            0.29240685, 0.3159288 ], 
           [0.24918965, 0.22611906, 0.23765829, 0.25470437, 0.27532474,
            0.28896069, 0.31268541]]

lr = [0.00001, 0.00002, 0.0001, 0.0005, 0.0015, 0.0025, 0.005, 0.01]
error_lr = [[0.3531271, 0.31796932, 0.25458222, 0.23081154, 0.2456418, 
             0.26853353, 0.29361466, 0.34443733],
            [0.35264225, 0.31708677, 0.25083991, 0.22382329, 0.24167567, 
             0.26616887, 0.29221101, 0.34400937]]

batch = [500, 1000, 2000, 5000, 10000]
error_batch = [[0.22405809, 0.23451039, 0.24321139, 0.26486551, 0.2908088],
               [0.21384077, 0.22466987, 0.2351169,  0.26025251, 0.28850835]]

default = [0.05, 0.0005, 5000]
titles = ['dropout', 'learning rate', 'batch size', 'num out']

fig1, ax1 = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
fig1.suptitle('Hyperparameters optimization \n .')

color = ['b', 'r']
labels = ['Validation', 'Training']

for i in range(2):
    ax1[0].plot(dropout, error_d[i], color=color[i], marker='x')    
    ax1[1].plot(lr, error_lr[i], color=color[i], marker='x')
    ax1[1].set_xscale('log')
    ax1[2].plot(batch, error_batch[i], color=color[i], marker='x')    
    
    if i==0:
        ax1[i].set_ylabel('Mean absolute error')

for j in range(3):
    ax1[j].set_xlabel(titles[j])
    ax1[j].set_title('Optimization of %s' %titles[j])
    ax1[j].yaxis.grid(True)
    ax1[j].set_ylim(0, 0.4)
    ax1[j].axvline(default[j], color='cornflowerblue')  
    
fig1.legend(labels)
plt.savefig("graphs/NN1_hyperopti_%s.pdf" %TESTNUM, dpi=150) 

#%% Influence number samples

ns = [1000, 5000, 10000, 50000, 100000, 200000, 400000]
n = len(ns)
error_ns = np.zeros((2,n))
x_train_old = x_train
target_train_old = target_train

# valid
x_valid_old = x_valid
target_valid_old = target_valid

for i in range(n):
    num_train = ns[i]
    x_train = x_train_old[0:num_train, :]
    target_train = target_train_old[0:num_train, :]
    
    #valid (checker si pas de difference)
    num_div = ns[i]
    x_valid = x_valid_old[0:num_div, :]
    target_valid = target_valid_old[0:num_div, :]   

    params1['batch_size'] = int(num_train/100)
    tic = time.time()
    trial = train_network1(params1)
    toc = time.time()
    train_time = toc - tic
    print("training time: ", train_time)
    
    error_ns[0, i] = trial['meanValError'][-1]
    error_ns[1, i] = trial['meanTrainError'][-1]
    
    print(error_ns[0,i], error_ns[1,i])
    
 
#%% Graph for number of samples

title = 'Influence of number of samples on learning'
ns = [1000, 5000, 10000, 50000, 100000, 200000, 400000]

error_ns = [[0.47650279, 0.39815512, 0.38194787, 0.28233113, 0.26564592,
             0.23772518, 0.23162131],
            [0.3807254,  0.34776414, 0.34402051, 0.25508525, 0.24806336,
             0.22598165, 0.22474477]]
fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

color = ['b', 'r']
labels = ['Validation', 'Training']

for i in range(2):
    ax1.plot(ns, error_ns[i], color=color[i], marker='x')    

ax1.set_ylabel('Mean absolute error')
ax1.set_xlabel('number of training samples')
ax1.set_title(title)
ax1.yaxis.grid(True)
ax1.set_ylim(0, 0.5)
    
fig1.legend(labels)
plt.savefig("graphs/NN1_numsamples.pdf", dpi=150) 



