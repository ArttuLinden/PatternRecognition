from os import getcwd
import numpy as np
from own_functions import loadData
import matplotlib.pyplot as plt
from scipy import signal

# Loading data
folder = getcwd() +'/robotsurface/'
data = loadData(folder)

# Choosing what to investigate
sampleInd = 16
sensorInd = 7


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

