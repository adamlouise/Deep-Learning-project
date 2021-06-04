#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 20:49:04 2021

@author: louiseadam
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import os
import sys

path_to_utils = os.path.join('.', 'python_functions')
path_to_utils = os.path.abspath(path_to_utils)

if path_to_utils not in sys.path:
    sys.path.insert(0, path_to_utils)

import mf_utils as util
import pickle

import time

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBRegressor

#%% DW-MRI Data

use_noise = True
small_set = False
big_set = True
num_params = 6
num_fasc = 2
M0 = 500
num_atoms = 782
num_sample = 600000
num_div = num_sample/6
n_test = '4_6'

save_boost = False
save_rf = False

if big_set:
    print("big_set")
    if use_noise:
        filename = 'synthetic_data/DW_noisy_store_uniform_600000__lou_version8'
        data = pickle.load(open(filename, 'rb'))
        data = data/M0
    else:
        filename = 'synthetic_data/DW_image_store_uniform_600000__lou_version8'
        data = pickle.load(open(filename, 'rb'))
    
    parameters = util.loadmat(os.path.join('synthetic_data',
                                            "training_datauniform_600000_samples_lou_version8"))    
    IDs = parameters['IDs'][0:num_sample, :]
    nus = parameters['nus'][0:num_sample, :]
        
if small_set:
    print("small_set")
    filename = 'data/ID_noisy_data_lownoise' 
    IDs_2 = pickle.load(open(filename, 'rb'))    
    filename = 'data/nus_data_lownoise' 
    nus_2 = pickle.load(open(filename, 'rb'))
    
    IDs = IDs_2[0:num_sample, :]
    nus = nus_2[0:num_sample, :]
    
    if use_noise:
        filename = 'data/dw_noisy_data_lownoise'
        data = pickle.load(open(filename, 'rb'))
    else:
        filename = 'data/dw_image_data_lownoise'
        data = pickle.load(open(filename, 'rb'))


use_dictionary = False # set to true if only IDs and nus are loaded (and not params)
if use_dictionary :
    data_dir = 'synthetic_data'
    parameters = util.loadmat(os.path.join(data_dir,
                                           "training_data_"
                                           "1000000_samples_safe.mat"))  
    
# divide data in train and test
x_train = data[:, 0:int(4*num_div)].T
x_test = data[:, int(4*num_div) : int(6*num_div) ].T


# %% Target data

print("--- Taking microstructural properties of fascicles ---")
        
target_params = np.zeros((6, num_sample))

target_params[0,:] = nus[:,0]
target_params[1,:] = parameters['subinfo']['rad'][IDs[:,0]]
target_params[2,:] = parameters['subinfo']['fin'][IDs[:,0]]
target_params[3,:] = nus[:,1]
target_params[4,:] = parameters['subinfo']['rad'][IDs[:,1]]
target_params[5,:] = parameters['subinfo']['fin'][IDs[:,1]]

# Standardisation
scaler1 = StandardScaler()
target_params = scaler1.fit_transform(target_params.T)
target_params = target_params.T

# Dividing in train test and valid
target_train = target_params[:, 0:int(num_div*4)].T
target_test = target_params[:, int(num_div*4) : int(6*num_div) ].T


#%% Gradient Boosting

# fit model to training data
tic = time.time()
boost = MultiOutputRegressor(XGBRegressor())
boost.fit(x_train, target_train)
toc = time.time()
t = toc - tic

# make predictions for test data
y = boost.predict(x_test)

# compute error
error_vect_b = abs(y - target_test)
error_b = np.mean(error_vect_b)

print('error boosting:', error_b)
print('time boosting:', t)

# Analysis of errors for 6 properties
error=[]
for i in range(6):
    mean_err = np.mean(error_vect_b[:,i])
    error.append(mean_err)
print(error)

if save_boost:
    filename = 'models_statedic/M3_GradientBoosting_version8_%s' %n_test 
    with open(filename, 'wb') as f:
            pickle.dump(boost, f)
            f.close()
            
# To load:        
# model_b = pickle.load(open(filename, 'rb'))
# model_b.get_params()


#%% Finding Max depth

use_DT = False  # test on decision tree
use_RF = True   # test on random forest
#essais = [2, 5, 15, 20]
max_depths = [12]
number_trees = [40]
errors_tree = []
times_tree = []
errors_rf = []
times_rf = []

for (i_md) in (max_depths):
    for j_nt in number_trees:
        print('--- max_depth:', i_md, ' - number of trees: ', j_nt)
        
        if use_DT:
            print('- DT')
            #For regression trees
            tic = time.time()
            regr = MultiOutputRegressor(DecisionTreeRegressor(max_depth=i_md))
            regr.fit(x_train, target_train)
            y = regr.predict(x_test)
            error_vect_tree = abs(y - target_test)
            error_tree = np.mean(error_vect_tree)
            toc = time.time()    
            t = toc - tic
            print('error tree:', error_tree)
            print('time tree:', t)
    
        if use_RF:
            print('- RF')
            #For random forest    
            tic = time.time()
            regr_rf = MultiOutputRegressor(RandomForestRegressor(n_estimators= j_nt, max_depth=i_md, random_state=0))
            regr_rf.fit(x_train, target_train)
            y = regr_rf.predict(x_test)
            error_vect_rf = abs(y - target_test)
            error_rf = np.mean(error_vect_rf)
            toc = time.time()
            t = toc - tic
            print('error rf:', error_rf)
            print('time rf:', t)

# save last rf model
if save_rf:
    filename = 'models_statedic/M3_RandomForest_version8_%s' %n_test 
    with open(filename, 'wb') as f:
            pickle.dump(regr_rf, f)
            f.close()

#%% Train property by property

for i in range(6):
    # fit model to training data
    tic = time.time()
    boost = XGBRegressor()
    boost.fit(x_train, target_train[:,i])
    toc = time.time()
    t = toc - tic
    
    # make predictions for test data
    y = boost.predict(x_test)
    
    error_b_prop = np.mean(abs(y - target_test[:,i]))
    
    print('error', error_b_prop)
    print('time', t)
    

#%%Influence number of samples

ns = [100000]

use_RF = False
use_B = True

estimators = [150, 300, 500, 700]
lr = [0.3, 0.4, 0.5]
max_depth = [4, 6, 8, 10, 12]

n = len(estimators)
error_ns = np.zeros((2,n))
time_ns = np.zeros((2,n))

hyperopti = True
k=0

for i in estimators:
    print(i)
    num_sample = ns[0]
    num_div = int(num_sample/4)
    
    # reshape data
    x_train = data[:, 0:int(4*num_div)].T
    x_test = data[:, int(4*num_div) : int(6*num_div) ].T
    target_train = target_params[:, 0:int(num_div*4)].T
    target_test = target_params[:, int(num_div*4) : int(6*num_div) ].T
    
    #For random forest  
    if use_RF:
        print('- RF')  
        tic = time.time()
        regr_rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=40, max_depth=12, random_state=0))
        regr_rf.fit(x_train, target_train)
        y = regr_rf.predict(x_test)
        error_vect_rf = abs(y - target_test)
        error_rf = np.mean(error_vect_rf)
        toc = time.time()
        t_rf = toc - tic
        print('error rf:', error_rf)
        print('time rf:', t_rf)
        
        error_ns[0, k] = error_rf
        time_ns[0, k] = t_rf
    
    #For boosting
    if use_B:
        print('- Boost') 
        tic = time.time()
        boost = MultiOutputRegressor(XGBRegressor())
        if hyperopti:
            boost = MultiOutputRegressor(XGBRegressor(
                                                    #learning_rate=i,
                                                    n_estimators=i,
                                                      #max_depth=i,
                                                      #min_sample_size = 0.8,
                                                      #early_stopping_rounds = 10,
                                                      #random_state=0,
                                                      verbosity = 1))
        else:
            boost = MultiOutputRegressor(XGBRegressor()) 
            
        boost.fit(x_train, target_train)
        output_b = boost.predict(x_test)
        error_vect_b = abs(output_b - target_test)
        error_b = np.mean(error_vect_b)
        toc = time.time()
        t_b = toc - tic
        print('error boosting:', error_b)
        print('time boosting:', t_b)
        
        error_ns[1, k] = error_b
        time_ns[1, k] = t_b
    
    print("Okayy -- ", error_ns[0,k], error_ns[1,k])
    k=k+1

print(error_ns)   


#%% graph ns

ns = [[500, 1000, 5000, 10000, 50000, 100000],
      [500, 1000, 5000, 10000, 50000, 100000, 200000]]
error_ns = [[0.48813444, 0.45192898, 0.39601538, 0.37363775, 0.34001969, 0.33068324],
            [0.49296739, 0.45386386, 0.38085599, 0.34661753, 0.28930647, 0.27714608, 0.27074564]]
time_ns = [[24.1441164, 59.53582883, 399.67495012, 971.5648458, 5643.99401307, 10663.51240182],
           [14.90619969, 30.28803015, 42.38691115, 821.09351707, 1509.23933291, 2760.05010104, 7083]]


xticks_ns =[[ 0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
            [ 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3]]
fig1, ax1 = plt.subplots(nrows=2, ncols=1, figsize=(7, 10))
fig1.suptitle('Influence of number of training samples')
color = ['forestgreen', 'goldenrod']
labels = ['RF', 'GBoost']

for i in range(2):
    ax1[0].plot(xticks_ns[i], error_ns[i], color=color[i], marker='x', label=labels[i])   
    ax1[1].plot(xticks_ns[i], time_ns[i], color=color[i], marker='x', label=labels[i])  

ax1[0].set_ylabel('Mean absolute error')
ax1[1].set_ylabel('Time for fitting')
ax1[1].set_xlabel('number of training samples')
ax1[0].set_title('Mean error')
ax1[1].set_title('Training time')
ax1[0].yaxis.grid(True)
ax1[1].yaxis.grid(True)
ax1[0].set_ylim(0, 0.6)
plt.setp(ax1, xticks=xticks_ns[1], xticklabels=ns[1])
    
fig1.legend(labels)
plt.savefig("graphs/M3_B_numsamples2.pdf", dpi=150) 


#%%

lr = [0.001, 0.005, 0.01, 0.015, 
      0.02, 0.05, 0.1, 0.2, 
      0.3, 0.4, 0.5]
error_lr = [0.8788570413305273, 0.6965119698666888, 0.5596090141501865, 0.48451220908939857,
            0.443604667336712, 0.37836974277730634, 0.36128754841889615, 0.3530009873275988,
            0.35223205470052044, 0.3568950652745298, 0.3645941250515524]

nest = [5, 15, 25, 50, 
        75, 100, 150, 200, 
        300, 500]
error_nest = [0.47805998314368303, 0.39451149451755585, 0.38031536771233915, 0.36572689125160335,
              0.35730623969761094, 0.35223205470052044, 0.34660292225324435, 0.34329164149688, 
              0.33972550348890185, 0.337535670651088]

maxdepth = [4, 6, 8, 10, 12]
error_maxdepth = [0.37164224, 0.35223205, 0.34776363, 0.35360352, 0.36169562]

xaxis = [lr, nest, maxdepth]
errors = [error_lr, error_nest, error_maxdepth]

titles=['Learning rate', 'Number of estimators', 'Maximum depth of trees']
xlabels = ['lr', 'n_estimators', 'max_depth']
default = [0.3, 100, 6]
fig2, ax2 = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
fig2.suptitle('Parameter tuning') 

for i in range(3):
    ax2[i].plot(xaxis[i], errors[i], color='goldenrod', marker='x')   
    ax2[i].axvline(default[i], color='cornflowerblue')   
    ax2[i].set_xlabel(xlabels[i])
    ax2[i].set_title(titles[i])
    ax2[i].yaxis.grid(True)
    ax2[i].set_ylim(0, 0.8)
    
ax2[0].set_ylabel('Mean absolute error')

plt.savefig("graphs/M3_B_params.pdf", dpi=150) 

