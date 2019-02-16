from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost import XGBClassifier as XGB
from sklearn.model_selection import GroupShuffleSplit,cross_val_score
from own_functions import getFeatures, loadData,getRelevantData,printScores,makeSubmissionFile

from os import getcwd
import numpy as np
folder = getcwd() +'/robotsurface/'
data = loadData(folder)
X = data[0]
y = data[1]
groups = data[2]
limit = 'all'
feature = 'mean_std'
X = getRelevantData(X,limit)
X_feats = getFeatures(X,feature,False,0)

params = {
        'max_depth':np.arange(2,10,1),
        'min_child_weight':[0.1,0.5,1,2],
        'gamma':[0,0.001,0.01,0.1,0.2],
        'learning_rate':[0.1,0.2,0.3,0.5,0.7,1]}

cv = GroupShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
clf = RandomizedSearchCV(XGB(n_jobs=-1), params, cv=cv)
clf.fit(X_feats,y,groups)

printScores(clf)

#%% test best found estimator
bestclf = clf.best_estimator_
bestclf.n_estimators = 200
testcv = GroupShuffleSplit(n_splits=30,test_size=0.2,random_state=0)
accuracies = cross_val_score(bestclf,X_feats,y,groups,cv=testcv)

print('Real score for best found estimator is {}'.format(
        np.mean(accuracies)))

#%% Make submission

chosenclf = XGB(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.2, max_delta_step=0,
       max_depth=7, min_child_weight=0.5, missing=None, n_estimators=200,
       n_jobs=-1, nthread=None, objective='multi:softprob', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)

limitmethod = 'vel_acc'
featuremethod = 'fftlog10'
aug_on = 1
normalize = False
pca_n = 0
predicions = makeSubmissionFile(chosenclf,data,limitmethod,featuremethod,aug_on,normalize,pca_n)



# temp test