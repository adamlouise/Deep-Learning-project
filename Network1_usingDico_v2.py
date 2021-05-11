#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:23:10 2020

@author: louiseadam

NW1 
"""

#%%
import numpy as np
#import matplotlib.pyplot as plt
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

# %% Train data

print('----------------------- Data --------------------------')

use_noise = True
print("Noise", use_noise)

num_params = 6
num_fasc = 2
M0 = 500
num_atoms = 782

if use_noise:
    filename = 'synthetic_data/DW_noisy_store_uniform_50000__lou_version5'
    y_data = pickle.load(open(filename, 'rb'))
    y_data = y_data/M0
    print('ok noise')
else:
    filename = 'synthetic_data/DW_image_store_uniform_50000__lou_version5'
    y_data = pickle.load(open(filename, 'rb'))
    print('ok no noise')

print(y_data[:5, :5])

#%%
M, num_sample = y_data.shape #M=552
#num_sample = 1000
num_div = num_sample/6
num_train = int(4*num_div)
num_test = int(num_train + num_div)
num_valid = int(num_test + num_div)

print('M', M) 
print('num_sample', num_sample)

# %% Back to data

# divide data in train, test and validation
x_train = y_data[:, 0:num_train]
x_test = y_data[:, num_train : num_test ]
x_valid = y_data[:, num_test : num_valid ]

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
x_valid = torch.from_numpy(x_valid)

print('x_train size', x_train.shape)
print('x_test size', x_test.shape)
print('x_valid size', x_valid.shape)

#quelques modifs pour le modele neuronal
x_train = x_train.float()
x_train = torch.transpose(x_train, 0, 1) 
x_test = x_test.float()
x_test = torch.transpose(x_test, 0, 1) 
x_valid = x_valid.float()
x_valid = torch.transpose(x_valid, 0, 1) 


# %% Target data

print("--- Taking microstructural properties of fascicles ---")

data_dir = 'synthetic_data'

target_data = util.loadmat(os.path.join('synthetic_data',
                                            "training_datauniform_50000_samples_lou_version5"))

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

print('target_params', target_params.shape)

## Standardisation

#print(target_params[:5, :5])

scaler1 = StandardScaler()
target_params = scaler1.fit_transform(target_params.T)
target_params = target_params.T

#print(target_params[:5, :5])

## Dividing in train test and valid
target_train = target_params[:, 0:num_train]
target_test = target_params[:, num_train : num_test ]
target_valid = target_params[:, num_test : num_valid ]


print('target_train size', target_train.shape)
#print('target_test size', target_test.shape)
print('target_valid size', target_valid.shape)

#quelques modifs pour le modele neuronal
target_train = torch.from_numpy(target_train).float()
target_train = torch.transpose(target_train, 0, 1) 
target_test = torch.from_numpy(target_test).float()
target_test = torch.transpose(target_test, 0, 1) 
target_valid = torch.from_numpy(target_valid).float()
target_valid = torch.transpose(target_valid, 0, 1) 

#%% Parameters

params1 = {
    #Training parameters
    "num_samples": num_sample,
     "batch_size": 500,  #2500
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
     #hp.choice(hsjdkfhs, )
}

# filename = 'model1_29_params1' 
# with open(filename, 'wb') as f:
#           pickle.dump(params1, f)
#           f.close()

# %% Building the network

class Net1(nn.Module):

    def __init__(self, num_in, num_h1, num_h2, num_h3, num_h4, num_h5, num_out, drop_prob):
        super(Net1, self).__init__()  
        # input layer
        self.W_1 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h1, num_in)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_h1), 0))
        self.l1_bn = nn.BatchNorm1d(num_h1)
        # hidden layer
        self.W_2 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h2, num_h1)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_h2), 0))
        self.l2_bn = nn.BatchNorm1d(num_h2)
        #second hidden layer
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
    
    print(net1)
    
    # Optimizer and Criterion
    optimizer = optim.Adam(net1.parameters(), lr=params1["learning_rate"], weight_decay=0.0000001)
    lossf = nn.MSELoss()

    print('----------------------- Training --------------------------')
    
    # setting hyperparameters and gettings epoch sizes
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
    
    # lambda function
    get_slice = lambda i, size: range(i * size, (i + 1) * size)
    
    for epoch in range(num_epochs):
        
        t = time.time() - start_time
        
        # Forward -> Backprob -> Update params
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

#%% Training

tic = time.time()
trial = train_network1(params1)  
toc = time.time()

print("training time:", toc-tic, "[sec]")
        
# filename = 'model1_withdico_louVersion1_4_NoNoise' 
# with open(filename, 'wb') as f:
#          pickle.dump(trial, f)
#          f.close()

# Specify a path
#PATH = "M1_version1_NoNoise_StateDict.pt"
#net = trial['model']

# Save
#torch.save(net.state_dict(), PATH)

# # Pour Load
# model = torch.load(PATH)
# model.eval()

# filename2 = 'scaler1_version1' 
# with open(filename2, 'wb') as f:
#          pickle.dump(scaler1, f)
#          f.close()
        
#%% Graphs for Learning

import matplotlib.pyplot as plt

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
        axs[i,j].axis([0, len(epoch), 0, 1])
        if j==0:
            axs[i,j].set_ylabel('Absolute Error')
        if i==0:
            axs[i,j].set_xlabel('Epochs')
        axs[i,j].set_title(labels[j] + ' for fascicle '+ str(i+1))
        axs[i,j].grid()
        #axs[i,j].legend(['Train error','Validation error'])
fig.legend(['Train error','Validation error'])       

## - 2 - Graoh for learning curve of Mean Error

plt.figure()
plt.plot(epoch, meanTrainError, 'r', epoch, meanValError, 'b')
plt.title('5000 samples')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
#plt.minorticks_on()
#plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.grid()
plt.legend(['Mean Train error','Mean Validation error'])
plt.xlabel('Epochs'), plt.ylabel('Mean Scaled Error')
plt.axis([0, len(epoch)-5, 0, 1])
plt.show()

## - 3 - Graph for Learning curve of 3 properties (mean over fascicles)

fig2, axs2 = plt.subplots(1, 3, sharey='row', figsize=(15,4))
fig2.suptitle('Learning curve for each property - mean over fascicles')
for j in range(3):    
    axs2[j].plot(epoch, (train_acc[:, j]+train_acc[:, j+3])/2, 'r', epoch, (valid_acc[:, j]+valid_acc[:, j+3])/2, 'b')
    axs2[j].axis([0, len(epoch), 0, 1])
    if j==0:
        axs2[j].set_ylabel('Absolute Error')
    axs2[j].set_xlabel('Epochs')
    axs2[j].set_title(labels[j])
    axs2[j].grid()

fig2.legend(['Train error','Validation error'])  


#%% Predictions
print('----------------------- Prediction --------------------------')

# Loading Network

# from Classes.Net1_Class import create_Net1   
# params1 = pickle.load(open('model1_29_params1', 'rb')) 
# net = create_Net1(params1)
# PATH = "M1_version1_Noise_StateDict.pt"
# net.load_state_dict(torch.load(PATH))
# net.eval()
      
# predict
  
net = trial['model']
output = net(x_test)
output = output.detach().numpy()

mean_err_scaled = np.zeros(6)
for i in range(6):
    mean_err_scaled[i] = mean_absolute_error(output[:,i], target_test[:,i])

properties = ['nu 1', 'rad 1', 'fin 1', 'nu 2', 'rad 2', 'fin 2']
plt.figure()
plt.bar(properties, mean_err_scaled)
 
# output = scaler1.inverse_transform(output)
# target_scaled = scaler1.inverse_transform(target_test)

# error = output - target_scaled

# abserror = abs(error)

# plt.figure()
# plt.plot(range(len(target_test)), error)
# plt.xlabel('samples')
# plt.ylabel('Abs error')
# plt.show()


# plt.figure()
# plt.title('distribution of r1 errors for triangular noise')
# plt.hist(abserror[:,1], density=False, bins=30)  # `density=False` would make counts
# plt.ylabel('Count')
# plt.xlabel('error on radius 1')
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.show()

# print(np.mean(abserror[:,1]))


#%% 95% interval

target_scaled = scaler1.inverse_transform(target_test)

error = output - target_scaled
conf_int = np.zeros(num_params)

for j in range(num_params):
    data = error[:,j]
    
    mean = np.mean(data)
    sigma = np.std(data)
    
    confint = stats.norm.interval(0.95, loc=mean, 
        scale=sigma)
    
    print(confint)

# %%Testing Optimisation

# trials = Trials()
# best = fmin(train_network1, params1, algo=tpe.suggest, max_evals=7,trials=trials)

# print(trials.best_trial['result']['loss'])

# n = len(trials.results)
# tomin = np.zeros(n)
# to_opti = np.zeros(n)
# for i in range(n):
#     tomin[i]= trials.results[i]['loss']
#     to_opti[i] = trials.results[i]['params']['learning_rate']

# plt.figure()
# plt.scatter(to_opti, tomin)
# plt.title('Influence of learning_rate (dropout=0.05, lr=0.001)')
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.xlabel('learning_rate'), plt.ylabel('Sum of errors')
# plt.show()

# filename = 'NW1_trials' 
# with open(filename, 'wb') as f:
#         pickle.dump(trials, f)
#         f.close()