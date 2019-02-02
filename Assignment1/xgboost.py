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

if __name__ == '__main__':
    folder = getcwd() +'/robotsurface/'
    data = loadData(folder)
    
    # Control reliability of evalutation vs. speed
    n_splits = 1
    
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
    # linear:   LDA(),LogR(),SGDC()
    # svm:      SVC(),LinearSVC()
    # ensemble: RFC(),XTree(),AdaB(),GradB()
    # neighbor: KNN(),RNeigh(),NearC()
    # neural:   MLPC()
    # other:    XGB()
     
#    clfs = [KNN(x) for x in np.arange(1,20,3)]
#    clf_names = [x.__class__.__name__+ str(x.n_neighbors) for x in clfs]
    
#    XGBOOST params - max_depth, min_child_weight, gamma
    clfs = [XGB(max_depth=x) for x in range(1,10)]+\
            [XGB(min_child_weight=x) for x in range(1,10)]+\
            [XGB(gamma=x) for x in np.linspace(0,1,10)]
    
    clf_names = ['XGB depth={} child={} gamma={:.2}'.format(
            x.max_depth,x.min_child_weight,float(x.gamma)) for x in clfs]
    
    
    
    
    
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
    np.save('xgb_scores',data)


#%% For submitting results
#clf = RFC(n_estimators = 100)
#limitmethod = 'vel_abs_acc_xy'
#featuremethod = 'mean_std'
#makeSubmissionFile(clf,data,le,limitmethod,featuremethod,1)
