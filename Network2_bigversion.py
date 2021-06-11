#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 11:23:10 2020

@author: louiseadam
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from torch.nn.parameter import Parameter

from sklearn.preprocessing import StandardScaler

#path_to_utils = os.path.join('.', 'python_functions')
#path_to_utils = os.path.abspath(path_to_utils)
#if path_to_utils not in sys.path:
#    sys.path.insert(0, path_to_utils)

import mf_utils as util
from hyperopt import fmin, tpe, STATUS_OK, Trials # package for optimization
from scipy import stats

from sklearn.metrics import mean_absolute_error

#%% Basic parameters

num_atoms = 782
num_fasc = 2
num_params = 6 #nombre de paramètres à estimer: ['nu1', 'r1 ', 'f1 ', 'nu2', 'r2 ', 'f2 ']

# Data
new_gen = False
nouvel_enregist = False
via_pickle = True
new_training = True

# Saving
save_params = False
save_scaler = False
save_net = False
save_res = False
TESTNUM = '9_6' # change name for not overwriting old graphs/data

params = {
    #Training parameters
    "num_samples": 600000,
     "batch_size": 5000,
     #"batch_size": hp.choice("batch_size", [500, 1000, 2000, 5000, 10000])
     "num_epochs": 35,
     
     #NW1 parameters
     #"num_w_out": hp.choice("num_w_out", [5, 10, 20, 30, 50, 100] ),
     "num_w_out": 50,
     "num_w_l1": 200,
     "num_w_l2": 600,
     "num_w_l3": 200,
     "num_w_l4": 50,
     
     #NW2
     "num_f_l1": 300,
     "num_f_l2": 200,
     "num_f_l3": 100,
     
     #other
     "learning_rate": 0.0001, 
     #"learning_rate": hp.uniform("learningrate", 0.0005, 0.01),
     #"learning_rate": hp.choice("learningrate", [0.0005, 0.0015, 0.0025, 0.0050, 0.0075, 0.0100, 0.0125, 0.0150]),
     "dropout": 0.1
     #"dropout": hp.choice("dropout", [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
}

if save_params:
    filename = 'params/M2_params_%s' %TESTNUM 
    with open(filename, 'wb') as f:
              pickle.dump(params, f)
              f.close()
              
# Numbers of train, test and validation

num_samples = params["num_samples"]
num_div = int(num_samples/6)

num_train_samples = int(num_div*4)
num_test_samples = int(num_train_samples + num_div)
num_valid_samples = int(num_test_samples + num_div)

if (num_samples != num_valid_samples):
    raise "Division of data in train, test, valid does not work"

# %% Data via pickle files

print("--- Load and reshape w vectors ---")

## Load 

filename1 = 'synthetic_data/dataNW2_w_store_version8'
filename2 = 'synthetic_data/dataNW2_targets_version8' 

if new_gen:   
    from getDataW import gen_batch_data
    w_store, target_params = gen_batch_data(0, num_samples, 'train')   
    if nouvel_enregist:
        with open(filename1, 'wb') as f:
                pickle.dump(w_store, f)
                f.close()
        with open(filename2, 'wb') as f:
                pickle.dump(target_params, f)
                f.close()
    print('generation OK via new_gen \n', 'saved?', nouvel_enregist)
    
if via_pickle:     
    w_store = pickle.load(open(filename1, 'rb'))
    target_params = pickle.load(open(filename2, 'rb'))   
    print('loading via pickle OK')

## Reshape data

# divide data in train, test and validation
x_train = np.zeros((num_train_samples, num_atoms, num_fasc))
x_test = np.zeros((num_div, num_atoms, num_fasc))
x_valid = np.zeros((num_div, num_atoms, num_fasc))

x_train[:, :, 0] = w_store[0:num_train_samples, 0:num_atoms]
x_train[:, :, 1] = w_store[0:num_train_samples, num_atoms: 2*num_atoms]

x_test[:, :, 0] = w_store[num_train_samples:(num_test_samples), 0:num_atoms]
x_test[:, :, 1] = w_store[num_train_samples:(num_test_samples), num_atoms: 2*num_atoms]

x_valid[:, :, 0] = w_store[(num_test_samples):num_samples, 0:num_atoms]
x_valid[:, :, 1] = w_store[(num_test_samples):num_samples, num_atoms: 2*num_atoms]

# changes for pytorch
x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
x_valid = torch.from_numpy(x_valid)

x_train = x_train.float()
x_test = x_test.float()
x_valid = x_valid.float()


# %% Target data

print("--- Take microstructural properties of fascicles ---")

# Scaling: Standardization of properties
scaler2 = StandardScaler()
target_params = scaler2.fit_transform(target_params) #scaler: (num_samples, num_features)
target_params = torch.from_numpy(target_params)

if save_scaler:
    filename = 'NN2_scaler2_version8_%s' %TESTNUM 
    with open(filename, 'wb') as f:
              pickle.dump(scaler2, f)
              f.close()

# Divide in train test and valid

target_train = target_params[:num_train_samples, :]
target_test = target_params[num_train_samples:num_test_samples, :]
target_valid = target_params[num_test_samples:num_samples, :]

target_train = target_train.float()
target_test = target_test.float()
target_valid = target_valid.float()


# %% Defining the networks

# Network 1
class Net_w(nn.Module):

    def __init__(self, num_in, num_h1, num_h2, num_h3, num_out, drop_prob):
        super(Net_w, self).__init__()  
        # input layer
        self.W_1 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h1, num_in)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_h1), 0))

        # hidden layers
        self.W_2 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h2, num_h1)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_h2), 0))
        
        self.W_3 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h3, num_h2)))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_h3), 0))
        
        self.W_4 = Parameter(init.kaiming_uniform_(torch.Tensor(num_out, num_h3)))
        self.b_4 = Parameter(init.constant_(torch.Tensor(num_out), 0))
        
        self.activation = torch.nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):

        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        x = self.dropout(x)

        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = F.linear(x, self.W_3, self.b_3)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = F.linear(x, self.W_4, self.b_4)

        return x


# Network 2
class Net_f(nn.Module):

    def __init__(self, num_in, num_h1, num_h2, num_out, drop_prob):
        super(Net_f, self).__init__()  
        # input layer
        self.W_1 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h1, num_in)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(num_h1), 0))

        # hidden layers
        self.W_2 = Parameter(init.kaiming_uniform_(torch.Tensor(num_h2, num_h1)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(num_h2), 0))
        
        self.W_3 = Parameter(init.kaiming_uniform_(torch.Tensor(num_out, num_h2)))
        self.b_3 = Parameter(init.constant_(torch.Tensor(num_out), 0))
        
        self.activation = torch.nn.ReLU()
        
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):

        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)
        x = self.dropout(x)
        x = F.linear(x, self.W_2, self.b_2)
        x = self.activation(x)
        x = self.dropout(x)
        x = F.linear(x, self.W_3, self.b_3)

        return x

# Total network to gather the two others
class Net_tot(nn.Module):

    def __init__(self, numw_in, numw_l1, numw_l2, numw_l3, numw_out, numf_in, numf_l1, numf_l2, numf_out, drop):
        super(Net_tot, self).__init__()  
        self.netw = Net_w(numw_in, numw_l1, numw_l2, numw_l3, numw_out, drop_prob=drop)
        self.netf = Net_f(numf_in, numf_l1, numf_l2, numf_out, drop_prob=drop)

    def forward(self, w1, w2):
        x1 = self.netw(w1)
        x2 = self.netw(w2)        
        x = torch.cat((x1, x2), axis=1)
        x = self.netf(x)

        return x

#%% wrap function for Training 

def train_network(params: dict):

    num_w_out = params["num_w_out"] 
    num_w_l1 = params["num_w_l1"]
    num_w_l2 = params["num_w_l2"]
    num_w_l3 = params["num_w_l3"]
    num_w_in = num_atoms
    num_f_out = num_params # number of parameters to estimate
    num_f_l1 = params["num_f_l1"]
    num_f_l2 = params["num_f_l2"]
    num_f_in = num_w_out*num_fasc
    drop = params["dropout"]
    
    net_tot = Net_tot(num_w_in, num_w_l1, num_w_l2, num_w_l3, num_w_out, num_f_in, num_f_l1, num_f_l2, num_f_out, drop)
    
    # Optimizer and Criterion
    optimizer = optim.Adam(net_tot.parameters(), 
                           lr=params["learning_rate"]
                           #weight_decay=0.0000001 #easier to tune parameters without weight decay
                           )
    lossf = nn.MSELoss()
    
    print('----------------------- Training --------------------------')
    
    start = time.time()
    
    # setting hyperparameters and gettings epoch sizes
    batch_size = params["batch_size"] 
    num_epochs = params["num_epochs"]
    
    num_samples_train = num_train_samples
    num_batches_train = num_samples_train // batch_size
    num_samples_valid = num_div
    num_batches_valid = num_samples_valid // batch_size
    
    # setting up lists for handling loss/accuracy
    train_acc = np.zeros((num_epochs, num_params))
    valid_acc = np.zeros((num_epochs, num_params))
    meanTrainError, meanValError  = [], []
    cur_loss = 0
    losses = []

    get_slice = lambda i, size: range(i * size, (i + 1) * size)
    
    for epoch in range(num_epochs): # Forward -> Backprob -> Update params

        ## Train
        cur_loss = 0
        net_tot.train()
        
        for i in range(num_batches_train):
            
            optimizer.zero_grad()
            slce = get_slice(i, batch_size)
    
            output = net_tot(x_train[slce, :, 0], x_train[slce, :, 1])
            
            # compute gradients given loss
            target_batch = target_train[slce]
            batch_loss = lossf(output, target_batch)
            batch_loss.backward()
            optimizer.step()
            
            cur_loss += batch_loss   
            
        losses.append(cur_loss / batch_size)
    
        net_tot.eval()
        
        ### Evaluate training
        train_preds = [[], [], [], [], [], []]
        train_targs = [[], [], [], [], [], []]
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            preds = net_tot(x_train[slce, :, 0], x_train[slce, :, 1])
            
            for j in range(num_params):
                train_targs[j] += list(target_train[slce, j].numpy())
                train_preds[j] += list(preds.data[:,j].numpy())
            
        ### Evaluate validation
        val_preds = [[], [], [], [], [], []]
        val_targs = [[], [], [], [], [], []]
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            preds = net_tot(x_valid[slce, :, 0], x_valid[slce, :, 1])
            
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
        
        #if epoch % 10 == 0:
        print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f, " %(
            epoch+1, losses[-1], meanTrainError[-1], meanValError[-1]))
        
    to_min = sum(valid_acc_cur)
    
    end = time.time()
    t = end-start
      
    return {"loss": to_min, 
            "model": net_tot, 
            "params": params, 
            "status": STATUS_OK,
            "train_acc": train_acc,
            "valid_acc": valid_acc,
            "meanTrainError": meanTrainError,
            "meanValError": meanValError,
            "time": t
            }

#%% Training the network 

trial = train_network(params)

net_tot = trial['model']
train_time = trial['time']

if save_net:
    PATH = "models_statedic/M2_version8_StateDict_%s.pt" %TESTNUM
    torch.save(net_tot.state_dict(), PATH)

if save_res:
    filename = 'results/M2_trial_version8_%s' %TESTNUM 
    with open(filename, 'wb') as f:
              pickle.dump(trial, f)
              f.close()


#%% Graphs for Learning

# If loading old results
if new_training==False:
    filename = 'results/NN2_trial_version8'
    trial = pickle.load(open(filename, 'rb'))


train_acc = trial['train_acc']
valid_acc = trial['valid_acc']
epoch = np.arange(params['num_epochs'])

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
plt.savefig("graphs/NN2_LC_6properties_%s.pdf" %TESTNUM, dpi=150)  

## - 2 - Graph for learning curve of Mean Error

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
plt.savefig("graphs/NN2_LC_MeanError_%s.pdf" %TESTNUM, dpi=150) 

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
plt.savefig("graphs/NN2_LC_3prop_%s.pdf" %TESTNUM, dpi=150) 

# %% Predictions with test data

# predict and time
output_test = net_tot(x_test[:,:,0], x_test[:,:,1])
output_test = output_test.detach().numpy()

# mean absolute scaled error for 6 properties
mean_err_scaled = np.zeros(6)
for i in range(6):
    mean_err_scaled[i] = mean_absolute_error(output_test[:,i], target_test[:,i])
    
# 95% interval
output_scaled = scaler2.inverse_transform(output_test)
target_scaled = scaler2.inverse_transform(target_test)

error = output_scaled - target_scaled
conf_int = np.zeros(num_params)

for j in range(num_params):
    data = error[:,j]   
    mean = np.mean(data)
    sigma = np.std(data)
    
    confint = stats.norm.interval(0.95, loc=mean, scale=sigma)
    
    print(confint)
    print((-confint[0]+confint[1])/2)
    

#%% Optimization of hyperparameters withOUT hyperopti package
# recompile hyperparameters dictionary between each test !!! 

# 1 # Opti dropout

#dropout = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
dropout= []
n = len(dropout)

error_d = np.zeros((2, n))
for i in range(n):
    params['dropout'] = dropout[i]
    tic = time.time()
    trial = train_network(params)
    #net_tot = trial['model']
    toc = time.time()
    train_time = toc - tic
    print("training time: ", train_time)
    
    error_d[0, i] = trial['meanValError'][-1]
    error_d[1, i] = trial['meanTrainError'][-1]
    
    print("Okayy -- ", error_d[0,i], error_d[1,i])

# 2 # Opti learning rate

#lr = [0.0005, 0.0015, 0.0025, 0.005, 0.01]
lr = [0.02]
n = len(lr)

error_lr = np.zeros((2, n))
for i in range(n):
    params['learning_rate'] = lr[i]
    tic = time.time()
    trial = train_network(params)
    #net_tot = trial['model']
    toc = time.time()
    train_time = toc - tic
    print("training time: ", train_time)
    
    error_lr[0, i] = trial['meanValError'][-1]
    error_lr[1, i] = trial['meanTrainError'][-1]
    
    print("Okayy -- ", error_lr[0,i], error_lr[1,i])

# 3 # Opti batch

#batch = [500, 1000, 2000, 5000, 10000]
batch = []
n = len(batch)

error_batch = np.zeros((2, n))
for i in range(n):
    params['batch_size'] = batch[i]
    tic = time.time()
    trial = train_network(params)
    #net_tot = trial['model']
    toc = time.time()
    train_time = toc - tic
    print("training time: ", train_time)
    
    error_batch[0, i] = trial['meanValError'][-1]
    error_batch[1, i] = trial['meanTrainError'][-1]
    
    print("Okayy -- ", error_batch[0,i], error_batch[1,i])

# 4 # Opti num out

#numout = [5, 10, 20, 30, 50, 100]
numout = []
n = len(numout)

error_numout = np.zeros((2, n))
for i in range(n):
    params['num_w_out'] = numout[i]
    tic = time.time()
    trial = train_network(params)
    #net_tot = trial['model']
    toc = time.time()
    train_time = toc - tic
    print("training time: ", train_time)
    
    error_numout[0, i] = trial['meanValError'][-1]
    error_numout[1, i] = trial['meanTrainError'][-1]
    
    print("Okayy -- ", error_numout[0,i], error_numout[1,i])

#%% Graphs Hyperopti

dropout = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
error_d = [[0.34716476, 0.3179273,  0.31890046, 0.32934551, 0.34542705, 
                0.35109823, 0.37680853],
               [0.27665024, 0.28168675, 0.296956,   0.31454883, 0.33441109,
                0.34237666, 0.37068665]]

# lr = [0.00001, 0.00005, 0.0001, 0.0005, 0.0015, 
#       0.0025, 0.005, 0.01, 0.1]
# error_lr = [[0.60567108, 0.4414315, 0.382302, 0.32422126, 0.31948587, 
#              0.3229621,  0.33137632, 0.38454634, 0.86579719],
#             [0.60722836, 0.44039448, 0.379057, 0.30708537, 0.29562791, 
#              0.29975651, 0.31639291, 0.37747079, 0.86664516]]

lr = [0.00001, 0.00005, 0.0001, 0.0005, 0.0015, 
      0.0025, 0.005, 0.01, 0.02]
error_lr = [[0.60567108, 0.4414315, 0.382302, 0.32422126, 0.31948587, 
             0.3229621,  0.33137632, 0.38454634, 0.463694],
            [0.60722836, 0.44039448, 0.379057, 0.30708537, 0.29562791, 
             0.29975651, 0.31639291, 0.37747079, 0.463258]]

batch = [500, 1000, 2000, 5000, 10000]
error_batch = [[0.3163683,  0.31043869, 0.3146933,  0.31794533, 0.33015395],
               [0.27685101, 0.27323786, 0.28391522, 0.29460461, 0.31509742]]

titles = ['dropout', 'learning rate', 'batch size', 'num out']
default = [0.1, 0.0015, 5000]
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
    ax1[j].set_ylim(0, 0.62)
    ax1[j].axvline(default[j], color='cornflowerblue') 
    
fig1.legend(labels)
plt.savefig("graphs/NN2_hyperopti_%s.pdf" %TESTNUM, dpi=150) 


#%% Influence number samples

ns = [1000, 5000, 10000, 50000]
n = len(ns)
error_ns = np.zeros((2,n))
x_train_old = x_train
target_train_old = target_train

for i in range(n):
    num_train_samples = ns[i]
    x_train = x_train_old[0:num_train_samples, :, :]
    target_train = target_train_old[0:num_train_samples, :]
    
    params['batch_size'] = int(num_train_samples/100)
    tic = time.time()
    trial = train_network(params)
    toc = time.time()
    train_time = toc - tic
    print("training time: ", train_time)
    
    error_ns[0, i] = trial['meanValError'][-1]
    error_ns[1, i] = trial['meanTrainError'][-1]
    
    print("Errors: ", error_ns[0,i], error_ns[1,i])

print(error_ns)    

#%% graph
   
title = 'Influence of number of samples on learning'
ns = [1000, 5000, 10000, 50000, 100000, 200000, 300000, 400000]
error_ns = [[0.555087998509407, 0.45636243124802905,  0.422436202565829, 
             0.3619969040155411, 0.34366337458292645, 0.3272846192121506,
              0.32049227754275006, 0.31839559972286224],
            [0.19053472578525543, 0.1817637433608373, 0.19458448886871338,
             0.24334178864955902, 0.248063363134861, 0.2731884519259135,
             0.2863716284434001, 0.2929443195462227]]

fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
color = ['b', 'r']
labels = ['Validation', 'Training']

for i in range(2):
    ax1.plot(ns, error_ns[i], color=color[i], marker='x')    

ax1.set_ylabel('Mean absolute error')
ax1.set_xlabel('number of training samples')
ax1.set_title(title)
ax1.yaxis.grid(True)
ax1.set_ylim(0, 0.6)
    
fig1.legend(labels)
plt.savefig("graphs/NN2_numsamples_%s.pdf" %TESTNUM, dpi=150) 
