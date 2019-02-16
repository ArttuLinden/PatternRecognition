from os import getcwd
import numpy as np
from own_functions import loadData
import matplotlib.pyplot as plt
from scipy import signal

# Loading data
folder = getcwd() +'/robotsurface/'
data = loadData(folder)


# Plotting example of every sensor data

channels = ['Orientation.X','Orientation.Y', 'Orientation.Z', 
'Orientation.W', 'AngularVelocity.X', 'AngularVelocity.Y', 'AngularVelocity.Z', 
'LinearAcceleration.X', 'LinearAcceleration.Y', 'LinearAcceleration.Z']

sampleInd = np.random.randint(1703)
sampleCategory = data[4][sampleInd][1]

fig = plt.figure(figsize=(12,12))
fig.suptitle(sampleCategory,fontsize=15)
plt.subplots_adjust(wspace=0.3,hspace=0.3)

for channelInd,channel in enumerate(channels):
    sampleData = data[0][sampleInd,channelInd,:]
    plt.subplot(5,2,sensorInd+1)
    plt.plot(sampleData)
    plt.title(channel)
fig.savefig('exampleData.png')

# %%
# Plotting examples of power spectral density, log scale to 
# remove dominance of  high magnitudes

categoryID = 7
categorySamples = data[0][data[1]==categoryID,:,:]
categoryName = data[4][data[1]==categoryID][1][1]

fig = plt.figure(figsize=(12,12))
plt.suptitle(categoryName,fontsize=15)
plt.subplots_adjust(wspace=0.3,hspace=0.3)

#sampleInds = np.random.randint(np.shape(categorySamples)[0],size=5)
sampleInds = [1]
for sampleInd in sampleInds:
    for channelInd,channel in enumerate(channels):
        if channelInd > 3:
            sample_freqs,PSD = signal.periodogram(categorySamples[sampleInd,channelInd])
            features = np.log10(PSD+1)
            plt.subplot(3,2,channelInd-3)
            plt.plot(sample_freqs,features)
            plt.title(channel)
fig.savefig('examplePSD.png')

# %% 
# Testing the effect of sampling frequency

categoryID = 7
categorySamples = data[0][data[1]==categoryID,:,:]
categoryName = data[4][data[1]==categoryID][1][1]

fig = plt.figure(figsize=(14,2))
sampleInds = [1]
for ind,fs in enumerate([0.1,1,10]):
    sample_freqs,PSD = signal.periodogram(categorySamples[sampleInd,4],fs=fs)
    features = np.log10(PSD+1)
    plt.subplot(1,3,ind+1)
    plt.plot(sample_freqs,features)
plt.show()


# %%
# Plotting principal component analysis

import numpy as np
np.load('pca_comparison.npy')

accuracies = [float(score) for score in all_scores[:,3]]

fig = plt.figure(figsize=(10,5))
plt.plot(accuracies)
plt.xlabel('Number of principal components')
plt.ylabel('Accuracy')
fig.savefig('PCA.png')





#%% Choosing what to investigate
sampleInd = 165
sensorInd = 3


# Plot original data
sampleData = data[0][sampleInd,sensorInd,:]
sampleCategory = data[4][sampleInd][1]
plt.figure(figsize=(10,5))
plt.plot(sampleData)
plt.title('Original data, '+sampleCategory)
plt.show()

# Plot fourier
features = np.abs(np.fft.fft(data[0])[:,:,:63])
samplefft = features[sampleInd,sensorInd,:]
plt.figure(figsize=(10,5))
plt.plot(samplefft)
plt.title('FFT, '+sampleCategory)

#%% Test plotting multiple sample of same category
categoryID = 6
sensorInd = 4
numSamples = 10
categorySamples = data[0][data[1]==categoryID,sensorInd,:]

features = np.abs(np.fft.fft(categorySamples)[:,:63])

plt.figure(figsize=(10,10))
plt.title('{} samples, {}'.format(numSamples,data[4][data[1]==categoryID][0,1]))
for plotInd in np.random.randint(0,np.shape(features)[0],size=numSamples):
    plt.plot(features[plotInd])


# %%Test finding peaks from fft
    
testData = samplefft

peaks = signal.find_peaks(testData,height=np.max(testData)*0.5)[0]
values = testData[peaks]
sorted_inds = np.argsort(values)[::-1]
end = np.min([3,len(sorted_inds)])

result = peaks[sorted_inds[:end]]

plt.figure(figsize=(10,5))
plt.plot(testData)
for peak in result:
    plt.axvline(x=peak,color='r',linewidth=1)
plt.title('Peaks fount at {}'.format(result))

