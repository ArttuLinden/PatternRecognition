# -*- coding: utf-8 -*-

# ==============================================================================
# Looping over:
#     - data limitation methods
#     - feature extraction methods
#     - classifiers
#==============================================================================

from os import getcwd
import numpy as np
from own_functions import loadData,testClassifier,testHyperopt
from hpsklearn import HyperoptEstimator,xgboost_classification,any_sparse_classifier
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
from hyperopt import hp

from sklearn.ensemble import RandomForestClassifier


folder = getcwd() +'/robotsurface/'
data = loadData(folder)


limit = 'vel_acc'

feats = 'fftlog10'

# Test hyperopt for finding optimal hyperparameters
clf = HyperoptEstimator(classifier=any_sparse_classifier('test'))
score,bst = testHyperopt(data,clf,limit,feats)

#%%
# Test if it really works


    
scores = testClassifier(data,bst['learner'],limit,feats,5)

print("{:.5f} accuracy for {}, {}, {}".format(np.mean(scores),limit,feats,bst['learner']))



#%% Test ligbm cv

train_set = lgb.Dataset(data[0], data[1], data[2])

model = lgb.LGBMClassifier()

# Discrete uniform distribution
num_leaves = {'num_leaves': hp.quniform('num_leaves', 30, 150, 1)}

# Learning rate log uniform distribution
learning_rate = {'learning_rate': hp.loguniform('learning_rate',
                                                 np.log(0.005),
                                                 np.log(0.2))}

# Define the search space
space = {
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'boosting_type': hp.choice('boosting_type', 
                               [{'boosting_type': 'gbdt', 
                                    'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                 {'boosting_type': 'dart', 
                                     'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                 {'boosting_type': 'goss'}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}

def objective(params, n_folds = N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""
    
    # Perform n_fold cross validation with hyperparameters
    # Use early stopping and evalute based on ROC AUC
    cv_results = lgb.cv(params, train_set, nfold = n_folds, num_boost_round = 10000, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)
  
    # Extract the best score
    best_score = max(cv_results['auc-mean'])
    
    # Loss must be minimized
    loss = 1 - best_score
    
    # Dictionary with information for evaluation
return {'loss': loss, 'params': params, 'status': STATUS_OK}
