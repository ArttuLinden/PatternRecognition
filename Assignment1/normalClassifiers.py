# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from os import getcwd
from pandas import read_csv,to_numeric
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
import numpy as np
import csv

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC,LinearSVC
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


#==============================================================================
# Loading data: X, y, groups, X_test_orig
#==============================================================================

folder = getcwd() +'/robotsurface/'
X_test_orig = np.load(folder+'X_test_kaggle.npy')
X = np.load(folder+'X_train_kaggle.npy')
y_orig = read_csv(folder+'y_train_final_kaggle.csv').values
le = LabelEncoder()
le.fit(y_orig[:,1])
y = le.transform(y_orig[:,1])
groups = to_numeric(read_csv(folder+'groups.csv').values[:,1])
data = [X,y,groups,X_test_orig]

#==============================================================================
# Function definitions
#==============================================================================

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

def test_classifier(data,clf,limitmethod,featuremethod,n_splits):
    X = data[0]
    X = getRelevantData(X,limitmethod)
    X_f = getFeatures(X,featuremethod)
    y = data[1]
    groups = data[2]
    
    cv = GroupShuffleSplit(n_splits=n_splits,test_size=0.2)
    return cross_val_score(clf,X_f,y,groups,cv=cv)

def makeSubmissionFile(clf,data,le,limitmethod,featuremethod,aug_on=0):
    X = data[0]
    y = data[1]
    X_test = data[3]
    X = getRelevantData(X,limitmethod)
    X_f = getFeatures(X,featuremethod)
    X_test = getRelevantData(X_test,limitmethod)
    X_test_f = getFeatures(X_test,featuremethod)
    clf.fit(X_f,y)
    y_pred = clf.predict(X_test_f)
    # Use predictions to fit to test data as well
    if aug_on == 1:
        X_augmented = np.vstack((X_f,X_test_f))
        y_augmented = np.hstack((y,y_pred))
        clf.fit(X_augmented,y_augmented)
        y_pred = clf.predict(X_test_f)
    
    labels =list(le.inverse_transform(y_pred))

    with open("submission.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["# Id","Surface"])
        for i, label in enumerate (labels):
            writer.writerow(["{}".format(i),"{}".format(label)])

# %%==============================================================================
# Looping over:
#     - data limitation methods
#     - feature extraction methods
#     - classifiers
#==============================================================================

# Control reliability of evalutation vs. speed
n_splits = 200

# Filter data:
# basic:    'orient','vel','acc','orient_vel','orient_acc','vel_acc','all'
# advanced: 'abs_acc_xy','vel_abs_acc_xy'
limit_methods = ['orient','vel','acc','orient_vel','orient_acc','vel_acc','all', \
                 'abs_acc_xy','vel_abs_acc_xy']

# Extract features:
# basic:    'mean','std','mean_std','all'
feature_methods = ['mean','std','mean_std','all']

# Classifiers:
# linear:   LDA(),LogR(),SGDC()
# svm:      SVC(),LinearSVC()
# ensemble: RFC(),XTree(),AdaB(),GradB()
# neighbor: KNN(),RNeigh(),NearC()
# neural:   MLPC()


classifiers = [MLPC(),LDA(),LogR(),SGDC(),SVC(),LinearSVC(),RFC(),XTree(),AdaB(), \
                GradB(),KNN()]

clf_names = [x.__class__.__name__ for x in classifiers]

all_scores = []
for limit_method in limit_methods:
    for feature_method in feature_methods:
        for clf,clf_name in zip(classifiers,clf_names):
            scores = test_classifier(data,clf,limit_method,feature_method,n_splits)
            all_scores.append([clf_name,limit_method,feature_method,scores])
            mean_score = np.mean(scores)
            print("{:.5f} accuracy for {}, {}, {}".format(
                mean_score,limit_method,feature_method,clf_name))


# Save data
data = []
for score in all_scores:
    data.append([score[0],score[1],score[2],np.mean(score[3])])
data = np.array(data)
np.save('new_scores',data)


#%% For submitting results
#clf = RFC(n_estimators = 100)
#limitmethod = 'vel_abs_acc_xy'
#featuremethod = 'mean_std'
#makeSubmissionFile(clf,data,le,limitmethod,featuremethod,1)
