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

#%% DW-MRI Data

print('----------------------- Data --------------------------')

use_noise = True
small_set = False
big_set = True
num_params = 6
num_fasc = 2
M0 = 500
num_atoms = 782

if big_set:
    print("big_set")
    print('data')
    if use_noise:
        filename = 'synthetic_data/DW_noisy_store__1000000__lou_version1'
        data = pickle.load(open(filename, 'rb'))
        data = data/M0
    else:
        filename = 'synthetic_data/DW_image_store__1000000__lou_version1'
        data = pickle.load(open(filename, 'rb'))
    
    print('target')
    parameters = util.loadmat(os.path.join('synthetic_data',
                                            "training_data_"
                                            "1000000_samples_lou_version1"))
        
if small_set:
    print("small_set")
    filename = 'data/ID_noisy_data_lownoise' 
    IDs_2 = pickle.load(open(filename, 'rb'))    
    filename = 'data/nus_data_lownoise' 
    nus_2 = pickle.load(open(filename, 'rb'))
    
    if use_noise:
        filename = 'data/dw_noisy_data_lownoise'
        data = pickle.load(open(filename, 'rb'))
    else:
        filename = 'data/dw_image_data_lownoise'
        data = pickle.load(open(filename, 'rb'))

#%% Reshaping for num_samples
#M, num_sample = data.shape #M=552
num_sample = 600000
num_div = num_sample/2

if big_set:
    IDs = parameters['IDs'][0:num_sample, :]
    nus = parameters['nus'][0:num_sample, :]
if small_set:
    IDs = IDs_2[0:num_sample, :]
    nus = nus_2[0:num_sample, :]
    
# divide data in train and test
x_train = data[:, 0:int(num_div)].T
x_test = data[:, int(num_div) : int(2*num_div) ].T

#print(x_train)
print('x_train size', x_train.shape)
print('x_test size', x_test.shape)

# %% Target data

print("--- Taking microstructural properties of fascicles ---")

data_dir = 'synthetic_data'
use_dictionary = False

if use_dictionary :
    if use_noise:
        parameters = util.loadmat(os.path.join(data_dir,
                                                    "training_data_triangSNR_"
                                                    "1000000_samples_safe.mat"))
    else:
        parameters = util.loadmat(os.path.join(data_dir,
                                                    "training_data_"
                                                    "1000000_samples_safe.mat"))  
        
target_params = np.zeros((6, num_sample))

target_params[0,:] = nus[:,0]
target_params[1,:] = parameters['subinfo']['rad'][IDs[:,0]]
target_params[2,:] = parameters['subinfo']['fin'][IDs[:,0]]
target_params[3,:] = nus[:,1]
target_params[4,:] = parameters['subinfo']['rad'][IDs[:,1]]
target_params[5,:] = parameters['subinfo']['fin'][IDs[:,1]]

print('target_params', target_params.shape)

## Standardisation

scaler1 = StandardScaler()
target_params = scaler1.fit_transform(target_params.T)
target_params = target_params.T

## Dividing in train test and valid
prop = 3 #between 0 and 5: the property we will predict
target_train = target_params[:, 0:int(num_div)].T
target_test = target_params[:, int(num_div) : int(2*num_div) ].T

print('target_train size', target_train.shape)
print('target_test size', target_test.shape)


#%% Decision tree

# Fit regression model
#regr_1 = DecisionTreeRegressor(max_depth=2)
#regr_2 = DecisionTreeRegressor(max_depth=5)
#regr_1.fit(x_train, target_train)
#regr_2.fit(x_train, target_train)

# Predict
#y_1 = regr_1.predict(x_test)
#y_2 = regr_2.predict(x_test)

# Plot the results

## Graohe nul

# print(x_train.shape, target_train.shape)
# plt.figure()
# plt.plot(range(100), target_test, color="red", label="target", linewidth=2)
# plt.plot(range(100), y_1, color="cornflowerblue",
#          label="max_depth=2", linewidth=2)
# plt.plot(range(100), y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()

# error1 = np.mean(abs(y_1 - target_test))
# error2 = np.mean(abs(y_2 - target_test))
# print("Trees:", error1, error2)


#%% Finding Max depth

use_DT = False
use_RF = True
#essais = [2, 5, 15, 20]
max_depths = [11]
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
            #errors_tree.append(error)
            #times_tree.append(t)
    
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
        
        # errors_rf.append(error)
        # times_rf.append(t)
        
    #regr_rf.feature_importances_

#%% Propriété par propriété

from xgboost import XGBRegressor

for i in range(6):
    # fit model no training data
    tic = time.time()
    boost = XGBRegressor()
    boost.fit(x_train, target_train[:,i])
    
    # make predictions for test data
    y = boost.predict(x_test)
    
    error_b_prop = np.mean(abs(y - target_test[:,i]))
    toc = time.time()
    t = toc - tic
    
    print('error', error_b_prop)
    print('time', t)
    
#%% Boosting trees

from xgboost import XGBRegressor

# fit model no training data
tic = time.time()
boost = MultiOutputRegressor(XGBRegressor())
boost.fit(x_train, target_train)

# make predictions for test data
y = boost.predict(x_test)

error_b = np.mean(abs(y - target_test))
toc = time.time()
t = toc - tic
print('error boosting:', error_b)
print('time boosting:', t)

#%% Analyser les 6 erreurs seules

error=[]
for i in range(6):
    mean_err = np.mean(error_vect_tree[:,i])
    error.append(mean_err)

print(error)


#%%  
plt.figure()
plt.plot(essais, errors_rf, marker='o')
plt.title('Errors - 50 000 training samples - SNR 80-100')
plt.xlabel('max depth')
plt.ylabel('mean absolute error')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()

plt.figure()
plt.plot(essais, times_rf, marker='o')
plt.title('Time - 50 000 training samples - SNR 80-100')
plt.xlabel('max depth')
plt.ylabel('Time')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()

plt.figure()
plt.plot(times_rf, errors_rf, marker='o')
plt.title('Time vs error for trees SNR 80-100')
plt.ylabel('mean absolute error (scaled)')
plt.xlabel('computation time [s]')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()

plt.figure()
plt.plot(essais, errors_tree, essais, errors_rf, marker='o')
plt.title('Errors - 50 000 training samples - SNR 80-100')
plt.xlabel('max depth')
plt.ylabel('mean absolute error')
plt.legend(['Trees','Random Forest'])
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()

plt.figure()
plt.plot(essais, times_tree, essais, times_rf, marker='o')
plt.title('Time - 50 000 training samples - SNR 80-100')
plt.xlabel('max depth')
plt.ylabel('Time')
plt.legend(['Trees','Random Forest'])
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()

plt.figure()
plt.plot(times_tree, errors_tree, marker='o')
plt.title('Time vs error for trees SNR 80-100')
plt.ylabel('mean absolute error (scaled)')
plt.xlabel('computation time [s]')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()

plt.figure()
plt.plot(errors_rf, times_rf, marker='o')
plt.title('Time vs error for RandomForest SNR 80-100')
plt.xlabel('mean absolute error')
plt.ylabel('computation time')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()

#%% Graph

# essais_plot = [2, 5, 7, 10, 12, 15, 17, 20, 25]
# errors_plot = [0.566, 0.458, 0.411 , 0.3618, 0.34208, 0.3266, 0.3226, 0.3206, 0.3203]
# times_plot = [555.2, 1020.5, 1687.1, 1991.3, 2548.4, 2502.71, 7798.3, 6898.4, 5664.4]

# plt.figure()
# plt.plot(essais_plot, errors_plot, marker='o')
# plt.xlabel('max depth')
# plt.ylabel('mean absolute error')
# plt.title('Error depending on tree size')
# plt.show()
# plt.figure()
# plt.plot(essais_plot, times_plot, marker='o')
# plt.xlabel('max depth')
# plt.ylabel('computation time')
# plt.title('Time depending on tree size')
# plt.show()
# plt.figure()
# plt.plot(errors_plot, times_plot, marker='o')
# plt.yscale("log")
# plt.xlabel('mean absolute error')
# plt.grid(b=True, which='major', color='#666666', linestyle='-')
# plt.minorticks_on()
# plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.ylabel('computation time')
# plt.title('Error vs Time')
# plt.show()


#%% 
a = [1, 2, 3, 4]
b = [2, 4, 6, 8]
c = [4, 6, 8, 10]
plt.figure()
plt.plot(a, b, c)
plt.xlabel('max depth')
plt.ylabel('Time')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.show()

#%% Save models

# filename = 'M3_RandomForest_2' 
# with open(filename, 'wb') as f:
#         pickle.dump(regr_rf, f)
#         f.close()
        
# filename = 'M3_GradientBoosting_2' 
# with open(filename, 'wb') as f:
#         pickle.dump(boost, f)
#         f.close()
        

    