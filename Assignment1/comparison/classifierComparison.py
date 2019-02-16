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
from sklearn.naive_bayes import GaussianNB

# Voting for combining classifiers
# from sklearn.ensemble import VotingClassifier
if __name__ == '__main__':
    folder = getcwd() +'/robotsurface/'
    data = loadData(folder)
    
    # Control reliability of evalutation vs. speed
    n_splits = 30
    
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
    # advanced:  'fftmean_fft_std','fft2peaks_fftstd_fftmean_mean_std','fftlog10' 
    feats = ['fftlog10pca30']
    
    # =============================================================================
    # CLASSIFIERS:
    # =============================================================================
    # linear:   LDA(),LogR(),SGDC()
    # svm:      SVC(),LinearSVC()
    # ensemble: RFC(),XTree(),AdaB(),GradB()
    # neighbor: KNN(),RNeigh(),NearC()
    # neural:   MLPC()
    # other:    XGB()
    # multiclass: OneVsRestClassifier(LinearSVC())
     
#    clfs = [KNN(x) for x in np.arange(1,20,3)] 
    
#    XGBOOST params - max_depth, min_child_weight, gamma
#    clfs = [XGB(gamma=x,n_estimators = 300) for x in [0.84,0.87]]
#    clf_names = ['XGB depth={} child={} gamma={:.2}'.format(
#        x.max_depth,x.min_child_weight,float(x.gamma)) for x in clfs]
    
    clfs = [LDA(),LogR(),SGDC(),
            LinearSVC(),
            MLPC(),
            OneVsRestClassifier(LinearSVC()),
            GaussianNB()]
    clf_names = [x.__class__.__name__ for x in clfs]
    
    # KNN with variable neighbors
    for n_neighbors in [5]:
        clfs.append(KNN(n_neighbors=n_neighbors))
        clf_names.append("KNeighborsClassifier ({} neighbors)".format(n_neighbors))
    
    # SVC with different kernels
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        clfs.append(SVC(kernel=kernel))
        clf_names.append("SVC ({})".format(kernel))
    
    # Ensemble with 100 and 200
    clfs_100 = [x(n_estimators=100) for x in [RFC,XTree,AdaB,GradB,XGB]]
    clfs.extend(clfs_100)
    clf_names.extend([x.__class__.__name__+" (100 estimators)" for x in clfs_100])
    
#    clfs_200 = [x(n_estimators=200) for x in [RFC,XTree,AdaB,GradB,XGB]]
#    clfs.extend(clfs_100)
#    clf_names.extend([x.__class__.__name__+"(200 estimators)" for x in clfs_100])
    
    
    
#    clfs = [LogR(C=x) for x in 10.0 ** np.arange(-4,3)]
#    clf_names = [x.__class__.__name__ for x in clfs]

    
#    clfs = [LDA(),LogR(),\
#            RFC(n_estimators=100),XTree(n_estimators=100),GradB(n_estimators=100),XGB(),\
#            KNN(n_neighbors=3),KNN(n_neighbors=5),KNN(n_neighbors=10)]
#    clf_names = [x.__class__.__name__ for x in clfs]
    
    
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
    np.save('classifier_comparison_fftlog10pca30',data)


#%% For submitting results
#clf = RFC(n_estimators = 100)
#limitmethod = 'vel_abs_acc_xy'
#featuremethod = 'mean_std'
#makeSubmissionFile(clf,data,le,limitmethod,featuremethod,1)
    
#clf = XGB(n_estimators=300,gamma = 0.87)
#limitmethod = 'vel_acc'
#featuremethod = 'fft2peaks_fftstd_fftmean_mean_std'
#makeSubmissionFile(clf,data,limitmethod,featuremethod,1)