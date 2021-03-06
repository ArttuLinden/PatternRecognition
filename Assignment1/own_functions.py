#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 08:58:44 2019

@author: tatu
"""

# TEST median filtering

from pandas import read_csv,to_numeric
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import csv
from scipy import signal
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA

def getFeatures(X,method,normalize,pca_n):
    # if 2D data, expand to 3D
    if len(np.shape(X)) == 2:
        X = np.expand_dims(X, axis=1)
    if method == 'all':
        # All sensors together - 1280 D
        X = np.reshape(X,[np.shape(X)[0],np.shape(X)[1]*np.shape(X)[2]])
    elif method == 'mean':
        # Mean over time axis - 10 D
        X = np.mean(X,2)
    elif method == 'std':
        # Standard deviation over time axis - 10 D
        X = np.std(X,2)
    elif method == 'mean_std':
        # Mean and std over time axis - 20 D
        X = np.hstack([np.mean(X,2),np.std(X,2)])
# =============================================================================
#         Fourier stuff
# =============================================================================
    elif method == 'PSD':
        sample_freqs,PSD = signal.periodogram(X)
        X = PSD.reshape([np.shape(PSD)[0],np.shape(PSD)[1]*np.shape(PSD)[2]])
    elif method == 'logPSD':
        sample_freqs,PSD = signal.periodogram(X)
        scaledPSD = np.log10(PSD+1)
        X = scaledPSD.reshape([np.shape(scaledPSD)[0],np.shape(scaledPSD)[1]*np.shape(scaledPSD)[2]])
    elif method == 'fft':
        # Fourier transform
        fft = np.abs(np.fft.fft(X))[:,:,:63]
        X = fft.reshape([np.shape(fft)[0],np.shape(fft)[1]*np.shape(fft)[2]])
    elif method == 'fftpow2':
        # Fourier to power 2
        fft = np.abs(np.fft.fft(X))[:,:,:63]**2
        X = fft.reshape([np.shape(fft)[0],np.shape(fft)[1]*np.shape(fft)[2]])
    elif method == 'fftlog10':
        # log of Fourier
        fft = np.log10(np.abs(np.fft.fft(X))[:,:,:63]+1)
        X = fft.reshape([np.shape(fft)[0],np.shape(fft)[1]*np.shape(fft)[2]])
    elif method == 'fftlog10_all':
        # data in fourier and normal combined
        fft = np.log10(np.abs(np.fft.fft(X))[:,:,:63]+1)
        result = np.vstack([fft.T,X.T]).T
        X = result.reshape([np.shape(result)[0],np.shape(result)[1]*np.shape(result)[2]])
    elif method == 'fftmean_fftstd':
        # Fourier combined with mean and std
        fft = np.abs(np.fft.fft(X))[:,:,:63]
        X = np.hstack([np.mean(fft,2),np.std(fft,2)])
    elif method == 'fftmean_fftstd':
        # Fourier combined with mean and std
        fft = np.log10(np.abs(np.fft.fft(X))[:,:,:63])
        X = np.hstack([np.mean(fft,2),np.std(fft,2)])
    elif method == 'mean_std_fftmean_fftstd':
        # Normal mean & std combined with Fourier mean std
        fft = np.abs(np.fft.fft(X))[:,:,:63]
        X = np.hstack([np.mean(X,2),np.std(X,2),np.mean(fft,2),np.std(fft,2)])
    elif method == 'max2fftpeaks':
        # Maximum 2 peaks in Fourier
        fft = np.abs(np.fft.fft(X))[:,:,:63]
        peaks = getMaxpeaks(fft,2)
        X = peaks.reshape([np.shape(peaks)[0],np.shape(peaks)[1]*np.shape(peaks)[2]])
    elif method == 'max3fftpeaks':
        # Maximum 3 peaks in Fourier
        fft = np.abs(np.fft.fft(X))[:,:,:63]
        peaks = getMaxpeaks(fft)
        X = peaks.reshape([np.shape(peaks)[0],np.shape(peaks)[1]*np.shape(peaks)[2]])
    elif method == 'mean_std_max3fftpeaks':
        fft = np.abs(np.fft.fft(X))[:,:,:63]
        mean = np.expand_dims(np.mean(X,2), axis=2)
        std = np.expand_dims(np.std(X,2), axis=2)
        peaks = getMaxpeaks(fft)
        result = np.vstack([mean.T,std.T,peaks.T]).T        
        X = result.reshape([np.shape(result)[0],np.shape(result)[1]*np.shape(result)[2]])
    elif method == 'fft2peaks_fftstd_fftmean':
        fft = np.abs(np.fft.fft(X))[:,:,:63]
        mean = np.expand_dims(np.mean(fft,2), axis=2)
        std = np.expand_dims(np.std(fft,2), axis=2)
        peaks = getMaxpeaks(fft,2)
        result = np.vstack([mean.T,std.T,peaks.T]).T        
        X = result.reshape([np.shape(result)[0],np.shape(result)[1]*np.shape(result)[2]])
    elif method == 'fft2peaks_fftstd_fftmean_mean_std':
        fft = np.abs(np.fft.fft(X[:,4:]))[:,:,:63]
        fftmean = np.mean(fft,2)
        fftstd = np.std(fft,2)
        mean = np.mean(X,2)
        std = np.std(X,2)
        peaks = getMaxpeaks(fft,2)
        peaks = peaks.reshape(np.shape(peaks)[0],np.shape(peaks)[1]*np.shape(peaks)[2])
        X = np.vstack([fftmean.T,fftstd.T,peaks.T,mean.T,std.T]).T        
# =============================================================================
# More tests
# =============================================================================
    elif method == 'fftlog1030PCA_mean_std':
        mean = np.mean(X,2)
        std = np.std(X,2)
        
        fft = np.log10(np.abs(np.fft.fft(X[:,4:]))[:,:,:63]+1)
        fft = fft.reshape([np.shape(fft)[0],np.shape(fft)[1]*np.shape(fft)[2]])
        pca = PCA(n_components=25)
        fft = pca.fit_transform(fft)
        
        X = np.vstack([fft.T,mean.T,std.T]).T
    elif method == 'maxPSD_mean_std':
        mean = np.mean(X,2)
        std = np.std(X,2)
        
        sample_freqs,PSD = signal.periodogram(X[:,4:])
        scaledPSD = np.log10(PSD+1)
        maxPSD = np.argmax(scaledPSD,2)        
        X = np.vstack([maxPSD.T,mean.T,std.T]).T
        
    else:
        raise ValueError('Unknown feature extraction method')
        
    if normalize == True:
        # Scaling to zero mean and unit variance
        X = StandardScaler().fit_transform(X.astype('float').T).T
    if pca_n >0:
        pca = PCA(n_components=pca_n)
        X = pca.fit_transform(X)
    return X

def getRelevantData(X,method):
#==============================================================================
#     Standard combinations
#==============================================================================
    if method == 'all':
        # Everything
        pass
    elif isinstance(method,int) == True:
        # Leave one channel out
        channels = [x for x in range(10) if x != int(method)]
        X = X[:,channels]
    elif method == 'orient':
        # Only orientation
        X = X[:,:4]
    elif method == 'vel':
        # Only angular velocity
        X = X[:,4:7]
    elif method == 'acc':
        # Only linear acceleration
        X = X[:,7:]
    elif method == 'orient_vel':
        # Orientation and velocity
        X = X[:,:7]
    elif method == 'orient_acc':
        # Orientation and acceleration
        X = X[:,[0,1,2,3,7,8,9]]
    elif method == 'vel_acc':
        # Velocity and acceleration
        X = X[:,4:]
#==============================================================================
# More advanced filtering
#==============================================================================
    elif method == 'abs_acc_xy':
        # Acceleration as absolute values in X and Y directions
        X = X[:,7:]
        X[:,:2] = np.abs(X[:,:2])
    elif method == 'vel_abs_acc_xy':
        # Velocity and Acceleration (absolute values in X and Y directions)
        X = X[:,4:]
        X[:,3:5] = np.abs(X[:,3:5])
    else:
        raise ValueError('Unknown data limiting method')
    return X

def testClassifier(data,clf,limitmethod,featuremethod,n_splits,normalize=False,pca_n=0):
    X = data[0]
    X = getRelevantData(X,limitmethod)
    X_f = getFeatures(X,featuremethod,normalize,pca_n)
    y = data[1]
    groups = data[2]
    
    cv = GroupShuffleSplit(n_splits=n_splits,test_size=0.2,random_state=0)
    return cross_val_score(clf,X_f,y,groups,cv=cv)

def testHyperopt(data,clf,limitmethod,featuremethod):
    X = data[0]
    X = getRelevantData(X,limitmethod)
    X_f = getFeatures(X,featuremethod)
    y = data[1]
    groups = data[2]
    
    cv = GroupShuffleSplit(n_splits=1,test_size=0.2)
    for train_inds,test_inds in cv.split(X,y,groups):
        clf.fit(X_f[train_inds],y[train_inds])
        score = clf.score(X_f[test_inds],y[test_inds])
        bst = clf.best_model()
        print(score)
        print(bst)
    return score,bst

def makeSubmissionFile(clf,data,limitmethod,featuremethod,aug_on=0,normalize=False,pca_n=0):
    X = data[0]
    y = data[1]
    X_test = data[3]
    X = getRelevantData(X,limitmethod)
    X_f = getFeatures(X,featuremethod,normalize,pca_n)
    X_test = getRelevantData(X_test,limitmethod)
    X_test_f = getFeatures(X_test,featuremethod,normalize,pca_n)
    clf.fit(X_f,y)
    y_pred = clf.predict(X_test_f)
    
# =============================================================================
# Data augmentation:
# =============================================================================
    # Use predictions to fit to test data as well
    # predictions that are > aug_limit sure will be used
    if aug_on == 1:
        aug_limit = 0.8
        print("Augmenting with test samples whose predictions are {}% sure."\
              .format(aug_limit*100))
        
        test_pred = clf.predict_proba(X_test_f)
        good_feats = X_test_f[(np.max(test_pred,1)>aug_limit),:]
        good_preds = y_pred[(np.max(test_pred,1)>aug_limit)]
        print("Augmented with {:.2f} % of the test samples".format(
                100*np.shape(good_preds)[0]/np.shape(y_pred)[0]))
        
        X_augmented = np.vstack((X_f,good_feats))
        y_augmented = np.hstack((y,good_preds))
        
        clf.fit(X_augmented,y_augmented)
        y_pred = clf.predict(X_test_f)
    
    le = data[5]
    labels =list(le.inverse_transform(y_pred))

    with open("submission.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["# Id","Surface"])
        for i, label in enumerate (labels):
            writer.writerow(["{}".format(i),"{}".format(label)])
            
#==============================================================================
# Loading data: X, y, groups, X_test_orig
#==============================================================================
def loadData(folder):
    X_test_orig = np.load(folder+'X_test_kaggle.npy')
    X = np.load(folder+'X_train_kaggle.npy')
    y_orig = read_csv(folder+'y_train_final_kaggle.csv').values
    le = LabelEncoder()
    le.fit(y_orig[:,1])
    y = le.transform(y_orig[:,1])
    groups = to_numeric(read_csv(folder+'groups.csv').values[:,1])
    data = [X,y,groups,X_test_orig,y_orig,le]
    return data

def getMaxpeaks(X,num_peaks=3):
    out = np.zeros([np.shape(X)[0],np.shape(X)[1],num_peaks])
    for a,sample in enumerate(X):
        for b,data in enumerate(sample):
            peaks = signal.find_peaks(data,height=np.max(data)*0.5)[0]
            if peaks.size > 0:
                values = data[peaks]
                sorted_inds = np.argsort(values)[::-1]
                end = np.min([num_peaks,len(sorted_inds)])
                out[a,b,:end] = peaks[sorted_inds[:end]]
    return out

def own_cross_val_score(clf,X,y,groups,cv):
    scores = []
    for train_index, test_index in cv.split(X,y,groups):
        # Split data to train and test set
        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]

        # Fitting and prediction
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test,y_pred)
        scores.append(score)
    return scores
                
def printScores(clf):
    results = clf.cv_results_
    sortinds = np.argsort(results['mean_test_score'])[::-1]
    sortedScores = np.array(results['mean_test_score'])[sortinds]
    sortedParams = np.array(results['params'])[sortinds]
    
    highlightString = "*"*(4+len(clf.estimator.__class__.__name__))
    print('\n{} results for\n\n* {} *\n{}'.format(
            clf.__class__.__name__,
            clf.estimator.__class__.__name__,
            highlightString))
    info = list(results['params'][0].keys())
    headline = "{:20} |"*(np.size(info)+1)
    textline = "{:20.6} |"+"{:20} | "*np.size(info)
    print(headline.format('Accuracy',*info))
    print("="*(np.size(info)+1)*23)
        
    for score,param in zip(sortedScores,sortedParams) :
        values = list(param.values())
        print(textline.format(score,*values))
        
    print("Best estimator:")
    print(clf.best_estimator_)
    
    
    #%%
#fft = np.abs(np.fft.fft(X[:,4:]))[:,:,:63]
#fftmean = np.mean(fft,2)
#fftstd = np.std(fft,2)
#mean = np.mean(X,2)
#std = np.std(X,2)
#peaks = getMaxpeaks(fft,2)
#peaks = peaks.reshape(np.shape(peaks)[0],np.shape(peaks)[1]*np.shape(peaks)[2])
#result = np.vstack([fftmean.T,fftstd.T,peaks.T,mean.T,std.T]).T        

    
    