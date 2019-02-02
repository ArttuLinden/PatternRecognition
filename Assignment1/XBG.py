# -*- coding: utf-8 -*-

# ==============================================================================
# Looping over:
#     - data limitation methods
#     - feature extraction methods
#     - classifiers
#==============================================================================

from os import getcwd
import numpy as np
from own_functions import loadData,testClassifier

from xgboost import XGBClassifier as XGB

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
    
    limits = ['all']
    
    # =============================================================================
    # FEATURES:
    # =============================================================================
    # basic:    'mean','std','mean_std','all'
    
    feats = ['mean_std']
    
    # =============================================================================
    # CLASSIFIERS:
    # =============================================================================
    # XGBOOST params - max_depth, min_child_weight, gamma
    
    clfs = [XGB(max_depth=x) for x in range(1,10)]+\
            [XGB(min_child_weight=x) for x in range(1,10)]+\
            [XGB(gamma=x) for x in np.linspace(0,1,10)]
    
    depths = np.arange(0,10)
    childs = np.arange(0,10)
    gammas = np.linspace(0,1,10)
    
    
    
    xgb_scores = []
    for a,b,c in [(a,b,c) for a in depths for b in childs for c in gammas]:
    
        clf = XGB(max_depth=a,min_child_weight=b,gamma=c)
        scores = testClassifier(data,clf,'all','mean_std',n_splits)
        xgb_scores.append([a,b,c,scores])
        
        print("{:.5f} accuracy for depth={} child={} gamma={:.2f}".format(
                np.mean(scores),a,b,c))
    
    
    # Save data
    data = []
    for score in xgb_scores:
        data.append([score[0],score[1],score[2],np.mean(score[3])])
    data = np.array(data)
    np.save('xgb_scores',data)


#%% For submitting results
#from os import getcwd
#import numpy as np
#from own_functions import loadData,testClassifier,makeSubmissionFile
#
#from xgboost import XGBClassifier as XGB
#from os import getcwd
#folder = getcwd() +'/robotsurface/'
#data = loadData(folder)
#
#bst = XGB(max_depth=3,min_child_weight=1,gamma=0.89)
#makeSubmissionFile(bst,data,'all','mean_std',1)
