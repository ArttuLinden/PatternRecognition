#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 19:07:39 2019

@author: tatu
"""

from os import getcwd
import numpy as np
from own_functions import loadData,getRelevantData,getFeatures,getMaxpeaks
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier as XGB
from xgboost import plot_importance
import matplotlib.pyplot as plt


folder = getcwd() +'/robotsurface/'
data = loadData(folder)

X = data[0]
X = getRelevantData(X,'all')
X_f = getFeatures(X,'mean_std')
y = data[1]
clf = XGB()
clf.fit(X_f,y)
# %%
plt.figure(figsize=(20,10))
ax = plt.axes()
plot_importance(clf,ax)
plt.show()

# %% see feature order

fft = np.abs(np.fft.fft(X))[:,:,:63]
fftmean = np.expand_dims(np.mean(fft,2), axis=2)
fftstd = np.expand_dims(np.std(fft,2), axis=2)
mean = np.expand_dims(np.mean(X,2), axis=2)
std = np.expand_dims(np.std(X,2), axis=2)
peaks = getMaxpeaks(fft,2)
result = np.vstack([fftmean.T,fftstd.T,peaks.T,mean.T,std.T]).T        
result = result.reshape([np.shape(result)[0],np.shape(result)[1]*np.shape(result)[2]])