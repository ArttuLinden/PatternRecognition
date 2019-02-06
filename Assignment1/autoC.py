from os import getcwd
import numpy as np
from own_functions import loadData,getFeatures,getRelevantData
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

from autosklearn.classification import AutoSklearnClassifier as AutoC

if __name__ == '__main__':
    folder = getcwd() +'/robotsurface/'
    data = loadData(folder)
    
    # Control reliability of evalutation vs. speed
    n_splits = 10
    
    # =============================================================================
    # FILTER:
    # =============================================================================
    # basic:    'orient','vel','acc','orient_vel','orient_acc','vel_acc','all'
    # advanced: 'abs_acc_xy','vel_abs_acc_xy'
    
    limits = ['vel_acc']
    
    # =============================================================================
    # FEATURES:
    # =============================================================================
    # basic:    'mean','std','mean_std','all'
    # advanced: 'fft','max3fftpeaks','mean_std_max3fftpeaks'
    
    feats = ['mean_std_max3fftpeaks']
    
    # =============================================================================
    # CLASSIFIERS:
    # =============================================================================
    # XGBOOST params - max_depth, min_child_weight, gamma
    
#    clfs = [XGB(max_depth=x) for x in range(1,10)]+\
#            [XGB(min_child_weight=x) for x in range(1,10)]+\
#            [XGB(gamma=x) for x in np.linspace(0,1,10)]
    
    X = data[0]
    X = getRelevantData(X,'vel_acc')
    X_f = getFeatures(X,'mean_std_max3fftpeaks')
    y = data[1]
    groups = data[2]
    
    scores = []
    clf = AutoC()
    cv = GroupShuffleSplit(n_splits=1,test_size=0.2)
    for train_index, test_index in cv.split(X_f,y,groups):
        # Split data to train and test set
        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]
        
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test,y_pred)
        scores.append(score)
    
    print("{:.5f} accuracy".format(np.mean(scores)))
    
