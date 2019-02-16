from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GroupShuffleSplit,cross_val_score
from own_functions import getFeatures, loadData,getRelevantData,printScores,makeSubmissionFile

from os import getcwd
import numpy as np
folder = getcwd() +'/robotsurface/'
data = loadData(folder)
X = data[0]
y = data[1]
groups = data[2]
limit = 'vel_acc'
feature = 'fftlog10'
X = getRelevantData(X,limit)
X_feats = getFeatures(X,feature,False,0)

params = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

cv = GroupShuffleSplit(n_splits=3,test_size=0.2,random_state=0)
clf = RandomizedSearchCV(RFC(n_jobs=-1,n_estimators=100), params, 
                         cv=cv,n_iter = 100)
clf.fit(X_feats,y,groups)

print(clf.best_estimator_)

printScores(clf)

#%% test best found estimator
bestclf = clf.best_estimator_
bestclf.n_estimators = 200
testcv = GroupShuffleSplit(n_splits=30,test_size=0.2,random_state=0)
accuracies = cross_val_score(bestclf,X_feats,y,groups,cv=testcv)

print('Real score for best found estimator is {}'.format(
        np.mean(accuracies)))


RFC()
#%% Make submission
RFC(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=80, max_features='sqrt', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=2000, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

limitmethod = 'vel_acc'
featuremethod = 'fftlog10'
aug_on = 1
normalize = False
pca_n = 0
predicions = makeSubmissionFile(chosenclf,data,limitmethod,featuremethod,aug_on,normalize,pca_n)