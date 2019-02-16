# -*- coding: utf-8 -*-

# ==============================================================================
# Looping over:
#     - data limitation methods
#     - feature extraction methods
#     - classifiers
#==============================================================================

from os import getcwd
import numpy as np
from own_functions import loadData,testClassifier,makeSubmissionFile

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC,LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as XTree
from sklearn.ensemble import AdaBoostClassifier as AdaB
from sklearn.ensemble import GradientBoostingClassifier as GradB
from sklearn.linear_model import LogisticRegression as LogR
from sklearn.linear_model import SGDClassifier as SGDC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import RadiusNeighborsClassifier as RNeigh
from sklearn.neighbors import NearestCentroid as NearC
from sklearn.neural_network import MLPClassifier as MLPC
from xgboost import XGBClassifier as XGB
from xgboost import plot_importance

folder = getcwd() +'/robotsurface/'
data = loadData(folder)

# Control reliability of evalutation vs. speed
n_splits = 30

# =============================================================================
# FILTER:
# =============================================================================
# basic:    'orient','vel','acc','orient_vel','orient_acc','vel_acc','all'
# advanced: 'abs_acc_xy','vel_abs_acc_xy'

limits = ['orient','vel','acc','orient_vel','orient_acc','vel_acc','all']

# =============================================================================
# FEATURES:
# =============================================================================
# basic:    'mean','std','mean_std','all'
# advanced:  'fftmean_fft_std','fft2peaks_fftstd_fftmean_mean_std','fftlog10' 
feats = ['mean_std']




clfs = [RFC(n_estimators=100)]
clf_names = [x.__class__.__name__ for x in clfs]


all_scores = []
for l,f,c in [(l,f,c) for l in limits for f in feats for c in zip(clfs,clf_names)]:

    scores = testClassifier(data,c[0],l,f,n_splits)
    all_scores.append([c[1],l,f,scores])
    
    print("{:.5f} accuracy for {}, {}, {}".format(np.mean(scores),l,f,c[1]))


# Save data
data = []
for score in all_scores:
    data.append([score[0],score[1],score[2],np.mean(score[3])])
data = np.array(data)
np.save('sensor_comparison',data)


#%% For submitting results
#clf = RFC(n_estimators = 100)
#limitmethod = 'vel_abs_acc_xy'
#featuremethod = 'mean_std'
#makeSubmissionFile(clf,data,le,limitmethod,featuremethod,1)
    
#clf = XGB(n_estimators=300,gamma = 0.87)
#limitmethod = 'vel_acc'
#featuremethod = 'fft2peaks_fftstd_fftmean_mean_std'
#makeSubmissionFile(clf,data,limitmethod,featuremethod,1)