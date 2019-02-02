#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 08:58:44 2019

@author: tatu
"""

from pandas import read_csv,to_numeric
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
import numpy as np
import csv

def getFeatures(X,method):
    # if 2D data, expand to 3D
    if len(np.shape(X)) == 2:
        X = np.expand_dims(X, axis=1)
            
    if method == 'all':
        # All sensors together - 1280 D
        X = np.reshape(X,[np.shape(X)[0],np.shape(X)[1]*np.shape(X)[2]])
    elif method == 'mean':
        # Mean over time axis - 10 D
        X = np.mean(X,2)
    elif method == 'std':
        # Standard deviation over time axis - 10 D
        X = np.std(X,2)
    elif method == 'mean_std':
        # Mean and std over time axis - 20 D
        X = np.hstack([np.mean(X,2),np.std(X,2)])
    else:
        raise ValueError('Unknown feature extraction method')
    return X

def getRelevantData(X,method):
#==============================================================================
#     Standard combinations
#==============================================================================
    if method == 'all':
        # Everything
        pass
    elif method == 'orient':
        # Only orientation
        X = X[:,:4]
    elif method == 'vel':
        # Only angular velocity
        X = X[:,4:7]
    elif method == 'acc':
        # Only linear acceleration
        X = X[:,7:]
    elif method == 'orient_vel':
        # Orientation and velocity
        X = X[:,:7]
    elif method == 'orient_acc':
        # Orientation and acceleration
        X = X[:,[0,1,2,3,7,8,9]]
    elif method == 'vel_acc':
        # Velocity and acceleration
        X = X[:,4:]
#==============================================================================
# More advanced filtering
#==============================================================================
    elif method == 'abs_acc_xy':
        # Acceleration as absolute values in X and Y directions
        X = X[:,7:]
        X[:,:2] = np.abs(X[:,:2])
    elif method == 'vel_abs_acc_xy':
        # Velocity and Acceleration (absolute values in X and Y directions)
        X = X[:,4:]
        X[:,3:5] = np.abs(X[:,3:5])
    else:
        raise ValueError('Unknown data limiting method')
    return X

def testClassifier(data,clf,limitmethod,featuremethod,n_splits):
    X = data[0]
    X = getRelevantData(X,limitmethod)
    X_f = getFeatures(X,featuremethod)
    y = data[1]
    groups = data[2]
    
    cv = GroupShuffleSplit(n_splits=n_splits,test_size=0.2)
    return cross_val_score(clf,X_f,y,groups,cv=cv)

def makeSubmissionFile(clf,data,limitmethod,featuremethod,aug_on=0):
    X = data[0]
    y = data[1]
    X_test = data[3]
    X = getRelevantData(X,limitmethod)
    X_f = getFeatures(X,featuremethod)
    X_test = getRelevantData(X_test,limitmethod)
    X_test_f = getFeatures(X_test,featuremethod)
    clf.fit(X_f,y)
    y_pred = clf.predict(X_test_f)
    
# =============================================================================
# Data augmentation:
# =============================================================================
    # Use predictions to fit to test data as well
    # predictions that are > aug_limit sure will be used
    if aug_on == 1:
        aug_limit = 0.8
        print("Augmenting with test samples whose predictions are {}% sure."\
              .format(aug_limit*100))
        
        test_pred = clf.predict_proba(X_test_f)
        good_feats = X_test_f[(np.max(test_pred,1)>aug_limit),:]
        good_preds = y_pred[(np.max(test_pred,1)>aug_limit)]
        print("Augmented with {:.2f} % of the test samples".format(
                100*np.shape(good_preds)[0]/np.shape(y_pred)[0]))
        
        X_augmented = np.vstack((X_f,good_feats))
        y_augmented = np.hstack((y,good_preds))
        
        clf.fit(X_augmented,y_augmented)
        y_pred = clf.predict(X_test_f)
    
    y_orig = data[4]
    le = LabelEncoder()
    le.fit(y_orig[:,1])
    labels =list(le.inverse_transform(y_pred))

    with open("submission.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["# Id","Surface"])
        for i, label in enumerate (labels):
            writer.writerow(["{}".format(i),"{}".format(label)])
            
#==============================================================================
# Loading data: X, y, groups, X_test_orig
#==============================================================================
def loadData(folder):
    X_test_orig = np.load(folder+'X_test_kaggle.npy')
    X = np.load(folder+'X_train_kaggle.npy')
    y_orig = read_csv(folder+'y_train_final_kaggle.csv').values
    le = LabelEncoder()
    le.fit(y_orig[:,1])
    y = le.transform(y_orig[:,1])
    groups = to_numeric(read_csv(folder+'groups.csv').values[:,1])
    data = [X,y,groups,X_test_orig,y_orig]
    return data