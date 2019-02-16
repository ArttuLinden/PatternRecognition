from os import getcwd
import numpy as np
from own_functions import loadData,getRelevantData,getFeatures
from sklearn.ensemble import RandomForestClassifier as RFC
from xgboost import XGBClassifier as XGB
from sklearn.linear_model import LogisticRegression as LogR
from sklearn.model_selection import GroupShuffleSplit,cross_val_score,train_test_split,GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from scipy import signal
import csv

def LogRandomF(X_train,X_test,y_train,y_test):
    clf = LogR()
    testPredictions = []
    predictions = []
    for channel in range(10):
        X_f = X_train[:,channel]
        X_test_f = X_test[:,channel]  
        clf.fit(X_f,y_train)
        predictions.append(clf.predict(X_f))
        testPredictions.append(clf.predict(X_test_f))
        
    _,X_train = signal.periodogram(X_train)
    X_train = np.log10(1+X_train)
    _,X_test = signal.periodogram(X_test)
    X_test = np.log10(1+X_test)
    
    for freq in range(65):
        X_f = X_train[:,:,freq]
        X_test_f = X_test[:,:,freq]  
        clf.fit(X_f,y_train)
        predictions.append(clf.predict(X_f))
        testPredictions.append(clf.predict(X_test_f))
    predictions = np.array(predictions).T
    
    clf = RFC(n_estimators=300)
    clf.fit(predictions,y_train)
    #Testing
    testPredictions = np.array(testPredictions).T
    return clf.predict(testPredictions)


def combinedTwo(X_train,X_test,y_train):
        
    # Predict with XGB for fftlog10
    clf1 = XGB(n_estimators=1000,gamma = 0.87)
    
    fft_train = np.log10(np.abs(np.fft.fft(X_train[:,4:]))[:,:,:63]+1)
    fft_train = fft_train.reshape([np.shape(fft_train)[0],np.shape(fft_train)[1]*np.shape(fft_train)[2]])
    
    fft_test = np.log10(np.abs(np.fft.fft(X_test[:,4:]))[:,:,:63]+1)
    fft_test = fft_test.reshape([np.shape(fft_test)[0],np.shape(fft_test)[1]*np.shape(fft_test)[2]])
    
    clf1.fit(fft_train,y_train)
    p1 = clf1.predict_proba(fft_test)
    
    # Predict with RF for mean_std
    clf2 = RFC(n_estimators = 1000)

    mean_std_train = np.hstack([np.mean(X_train,2),np.std(X_train,2)])
    mean_std_test = np.hstack([np.mean(X_test,2),np.std(X_test,2)])
    
    clf2.fit(mean_std_train,y_train)
    p2 = clf2.predict_proba(mean_std_test)
    
    # Take the prediction of which one of the classifiers is most sure of
    p = np.stack([p1,p2])
    predicted_classes = np.argmax(np.max(p,0),1)
    return predicted_classes

def baseline(X_train,X_test,y_train):
    # Predict with XGB for fftlog10
    clf1 = XGB(n_estimators=300,gamma = 0.87)
    
    fft_train = np.log10(np.abs(np.fft.fft(X_train[:,4:]))[:,:,:63]+1)
    fft_train = fft_train.reshape([np.shape(fft_train)[0],np.shape(fft_train)[1]*np.shape(fft_train)[2]])
    
    fft_test = np.log10(np.abs(np.fft.fft(X_test[:,4:]))[:,:,:63]+1)
    fft_test = fft_test.reshape([np.shape(fft_test)[0],np.shape(fft_test)[1]*np.shape(fft_test)[2]])
    
    clf1.fit(fft_train,y_train)
    return clf1.predict(fft_test)
     

folder = getcwd() +'/robotsurface/'
data = loadData(folder)
# Training
X = data[0]
y = data[1]
groups = data[2]

# Features
#X = np.hstack([np.mean(X,2),np.std(X,2)])
results = []
cv = GroupKFold(n_splits=10)
for [train_ind,test_ind] in cv.split(X,y,groups):

    X_train = X[train_ind]
    X_test = X[test_ind]
    y_train = y[train_ind]
    y_test = y[test_ind]
    
#    clf = RFC(n_estimators=100)
#    clf.fit(X_train,y_train)
#    y_pred = clf.predict(X_test)
    y_pred = baseline(X_train,X_test,y_train,y_test)  
#    y_pred = combinedTwo(X_train,X_test,y_train,y_test)   
#    y_pred = LogRandomF(X_train,X_test,y_train,y_test)
    score = accuracy_score(y_test,y_pred)
    print(score)
    results.append(score)

mean_score = np.mean(np.array(results))
print("Mean accuracy is {}".format(mean_score))
    
resultsCombined = [ 0.38235294117647056,0.7411764705882353,
                   0.8411764705882353,0.7602339181286549,
                   0.4647058823529412,0.7660818713450293,
                   0.26900584795321636,0.5588235294117647,
                   0.8128654970760234,0.44970414201183434]



def makeSubmissionFile(data):
    X = data[0]
    y = data[1]
    X_test = data[3]
    
# =============================================================================
#     Method
# =============================================================================
    y_pred = combinedTwo(X,X_test,y)
    
    le = data[5]
    labels =list(le.inverse_transform(y_pred))

    with open("submission.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["# Id","Surface"])
        for i, label in enumerate (labels):
            writer.writerow(["{}".format(i),"{}".format(label)])
