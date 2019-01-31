# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from pandas import read_csv,to_numeric
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LogisticRegression as LogR
from sklearn.linear_model import SGDClassifier as SGDC


#==============================================================================
# Loading data: X, y, groups, X_test_orig
#==============================================================================

folder = 'C:\\\\Users\\makine42\\Desktop\\PatternRecognition\\robotsurface\\'
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

def makeFeatureMatrix(X,method):
    if method == 'all':
        # All sensors together - 1280 D
        X = np.reshape(X,[np.shape(X)[0],np.shape(X)[1]*np.shape(X)[2]])
    elif method == 'mean':
        # Mean over time axis - 10 D
        X = np.mean(X,2)
    elif method == 'mean_std':
        # Mean and std over time axis - 20 D
        X = np.hstack([np.mean(X,2),np.std(X,2)])
    else:
        raise ValueError('Unknown feature extraction method')
    return X

def getRelevantData(X,method):
    if method == 'all':
        pass
    elif method == 'acc':
        # Only acceleration
        X = X[:,4:7]
    elif method == 'abs_acc_xy':
        # Acceleration as absolute values in X and Y directions
        X = X[:,4:7]
        X[:,4:6] = np.abs(X[:,4:6])
    else:
        raise ValueError('Unknown data limiting method')
    return X

def test_classifier(data,clf,limitmethod,featuremethod):
    X = data[0]
    X = getRelevantData(X,limitmethod)
    X_f = makeFeatureMatrix(X,featuremethod)
    y = data[1]
    groups = data[2]
    
    cv = GroupShuffleSplit(n_splits=20,test_size=0.2)
    return cross_val_score(clf,X_f,y,groups,cv=cv)

def makeSubmissionFile(clf,data,le,limitmethod,featuremethod):
    X = data[0]
    y = data[1]
    X_test = data[3]
    X, X_test = getRelevantData(X, X_test,limitmethod)
    X, X_test = makeFeatureMatrix(X,X_test,featuremethod)
    clf.fit(X,y)
    y_pred = clf.predict(X_test)
    labels =list(le.inverse_transform(y_pred))

    with open("submission.csv", "w") as fp:
        fp.write("# Id,Surface\\n")
        for i, label in enumerate (labels):
            fp.write("%d,%s\\n" % (i, label))

#==============================================================================
# Looping over:
#     - data limitation methods
#     - feature extraction methods
#     - classifiers
#==============================================================================

limit_methods = ['abs_acc_xy']

feature_methods = ['mean_std']

classifiers = [LDA(),
               LinearSVC(), # fails to converge
               SGDC(),
               SVC(),
               RFC(n_estimators=100),
               RFC(n_estimators=300),
               LogR()]

print("Results\n==========================")
all_scores = []
for limit_method in limit_methods:
    for feature_method in feature_methods:
        for clf in classifiers:
            scores = test_classifier(data,clf,limit_method,feature_method)
            all_scores.append([limit_method,feature_method,scores])
            mean_score = np.mean(scores)
            print("{:.5f} accuracy for {}, {}, {}".format(
                mean_score,limit_method,feature_method,clf.__class__.__name__))

# For submitting results
"""
chosen_clf = classifiers[np.argmax(mean_scores)]
print("Chosen classifier is {} with mean accuracy {:.5}".format(
        chosen_clf.__class__.__name__, np.max(mean_scores)))

makeSubmissionFile(clf,data,le,limitmethod,featuremethod)
"""