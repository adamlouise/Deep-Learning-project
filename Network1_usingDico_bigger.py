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

# %% Train data

print('----------------------- Data --------------------------')

use_noise = True

num_params = 6
num_fasc = 2
M0 = 500
num_atoms = 782

if use_noise:
    filename = 'synthetic_data/DW_noisy_store__1000000__lou_version1'
    y_data = pickle.load(open(filename, 'rb'))
else:
    filename = 'synthetic_data/DW_image_store__1000000__lou_version1'
    y_data = pickle.load(open(filename, 'rb'))

y_data = y_data/M0

M, size_data = y_data.shape #M=552
num_sample = size_data
num_div = num_sample/4

print('M', M) 
print('num_sample', num_sample)

# %% Back to data

# divide data in train, test and validation
x_train = y_data[:, 0:int(2*num_div)]
x_test = y_data[:, int(2*num_div) : int(3*num_div) ]
x_valid = y_data[:, int(3*num_div) : int(4*num_div) ]

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
                                            "training_data_"
                                            "1000000_samples_lou_version1"))

# Substrate (=fingerprint) properties
IDs = target_data['IDs']
nus = target_data['nus']
    
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
target_train = target_params[:, 0:int(2*num_div)]
target_test = target_params[:, int(2*num_div) : int(3*num_div) ]
target_valid = target_params[:, int(3*num_div) : int(4*num_div) ]


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
     "batch_size": 5000,  
     "num_epochs": 25,
     
     #NW2
     "num_h1": 200,
     "num_h2": 600,
     "num_h3": 1200,
     "num_h4": 600,
     "num_h5": 500,
     "num_h6": 400,
     "num_h7": 100,
     
     #other
     "learning_rate": 0.0005,
     #"learning_rate": hp.choice("learningrate", [0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002]),
     "dropout": 0.05
     #"dropout": hp.uniform("dropout", 0, 0.4)
     #hp.choice(hsjdkfhs, )
}


# %% Building the network

class Net1(nn.Module):

    def __init__(self, num_in, num_h1, num_h2, num_h3, num_h4, num_h5, num_h6, num_h7, num_out, drop_prob):
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
        
        self.W_6 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h6, num_h5)))
        self.b_6 = Parameter(init.constant_(torch.Tensor(num_h6), 0))
        
        self.W_7 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h7, num_h6)))
        self.b_7 = Parameter(init.constant_(torch.Tensor(num_h7), 0))
        
        self.W_8 = Parameter(init.kaiming_uniform_(torch.Tensor(num_out, num_h7)))
        self.b_8 = Parameter(init.constant_(torch.Tensor(num_out), 0))
        
        
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
        x = self.activation(x)
        x = self.dropout(x)
        
        x = F.linear(x, self.W_7, self.b_7)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = F.linear(x, self.W_8, self.b_8)

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
    num_h6 = params1["num_h6"]
    num_h7 = params1["num_h7"]
    drop_prob = params1["dropout"]
    
    net1 = Net1(num_in, num_h1, num_h2, num_h3, num_h4, num_h5, num_h6, num_h7, num_out, drop_prob)
    
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
        
filename = 'model1_withdico_louVersion1_big1' 
with open(filename, 'wb') as f:
         pickle.dump(trial, f)
         f.close()
        
#%% Graphs

import matplotlib.pyplot as plt

train_acc = trial['train_acc']
valid_acc = trial['valid_acc']
epoch = np.arange(params1['num_epochs'])

mean_train_error = trial['meanTrainError']

for j in range(6):    
    plt.figure()
    plt.plot(epoch, train_acc[:, j], 'r', epoch, valid_acc[:, j], 'b')
    plt.legend(['Train error','Validation error'])
    plt.xlabel('Updates'), plt.ylabel('Error')
    plt.show()
    
meanTrainError = trial['meanTrainError']
meanValError = trial['meanValError']

# Mean Error
print(trial['meanTrainError'])
plt.figure()
plt.plot(epoch, meanTrainError, 'r', epoch, meanValError, 'b')
plt.title('Learning Curve: Mean Error - DL after NNLS')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend(['Mean Train error','Mean Validation error'])
plt.xlabel('Epochs'), plt.ylabel('Mean Scaled Error')
plt.axis([0, 40, 0, 1])
plt.show()


#%% Predictions
# print('----------------------- Prediction --------------------------')

# net = trial['model']

# print(x_test.shape)
# output = net(x_test)
# output = output.detach().numpy()

# mean_err_scaled = np.zeros(6)
# for i in range(6):
#     mean_err_scaled[i] = mean_absolute_error(output[:,i], target_test[:,i])

# print("mean_abs_err", mean_err_scaled)

# properties = ['nu1', 'r1', 'f1', 'nu2', 'r2', 'f2']
# plt.figure()
# plt.bar(properties, mean_err_scaled)

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

# conf_int = np.zeros(num_params)

# for j in range(num_params):
#     data = error[:,j]
    
#     mean = np.mean(data)
#     sigma = np.std(data)
    
#     confint = stats.norm.interval(0.95, loc=mean, 
#         scale=sigma)
    
#     print(confint)

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