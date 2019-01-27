# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from pandas import read_csv,to_numeric
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LogisticRegression as LogR
from sklearn.linear_model import SGDClassifier as SGDC

"""
Loading data: X, y, groups
"""
folder = '/home/tatu/Desktop/PatternRecognition/data/'
X_test_orig = np.load(folder+'X_test_kaggle.npy')
X = np.load(folder+'X_train_kaggle.npy')
y_orig = read_csv(folder+'y_train_final_kaggle.csv').values
le = LabelEncoder()
le.fit(y_orig[:,1])
y = le.transform(y_orig[:,1])
groups = to_numeric(read_csv(folder+'groups.csv').values[:,1])
data = [X,y,groups,X_test]

def makeFeatureMatrix(X,Y,method):
    if method == 'all':
        # All sensors together - 1280 D
        X = np.reshape(X,[np.shape(X)[0],np.shape(X)[1]*np.shape(X)[2]])
        Y = np.reshape(Y,[np.shape(Y)[0],np.shape(Y)[1]*np.shape(Y)[2]])
    elif method == 'mean':
        # Mean over time axis - 10 D
        X = np.mean(X,2)
        Y = np.mean(Y,2)
    elif method == 'mean_std':
        # Mean and std over time axis - 20 D
        X = np.hstack([np.mean(X,2),np.std(X,2)])
        Y = np.hstack([np.mean(Y,2),np.std(Y,2)])
    else:
        raise ValueError('Unknown feature extraction method')
    return X,Y

def getRelevantData(X,Y,method):
    if method == 'all':
        pass
    elif method == 'acc':
        X = X[4:6]
        Y = Y[4:6]
    else:
        raise ValueError('Unknown data limiting method')
    return X,Y

def test_classifier(data,clf,limitmethod,featuremethod):
    X = data[0]
    y = data[1]
    groups = data[2]
    cv = ShuffleSplit(n_splits=10,test_size=0.5)
    scores = []
    for train_index, test_index in cv.split(X,y,groups):
        # Split data to train and test set
        X_train = X[train_index]
        y_train = y[train_index]
        print (np.shape(train_index))
        print (np.shape(test_index))
        X_test = X[test_index]
        y_test = y[test_index]
        # Taking relevant information
        X_train, X_test = getRelevantData(X_train, X_test,limitmethod)
        # Extracting features
        X_train_feat,X_test_feat = makeFeatureMatrix(X_train,X_test,featuremethod)
        # Fitting and prediction
        clf.fit(X_train_feat,y_train)
        y_pred = clf.predict(X_test_feat)
        score = accuracy_score(y_test,y_pred)
        scores.append(score)
    return scores

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
        fp.write("# Id,Surface\n")
        for i, label in enumerate (labels):
            fp.write("%d,%s\n" % (i, label))

"""
TO DO:
Transforming data into better format - absolute acc,velocity?
"""

limitmethod = 'all'
featuremethod = 'mean_std'

"""
classifiers = [LDA(),
               LinearSVC(tol=1), # fails to converge
               SGDC(max_iter=1000,tol=1e-3),
               SVC(gamma='scale'),
               RFC(n_estimators=100),
               LogR(solver='liblinear',multi_class='auto')]
"""
classifiers = [RFC(n_estimators=100)]

all_scores = []
for clf in classifiers:
    scores = test_classifier(data,clf,limitmethod,featuremethod)
    all_scores.append(scores)
    print("Mean accuracy for {} is {:.5}".format(
        clf.__class__.__name__, np.mean(scores)))

"""
chosen_clf = classifiers[np.argmax(mean_scores)]
print("Chosen classifier is {} with mean accuracy {:.5}".format(
        chosen_clf.__class__.__name__, np.max(mean_scores)))

makeSubmissionFile(clf,data,le,limitmethod,featuremethod)
"""