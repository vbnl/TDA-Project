#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 22:03:31 2020

@author: nathanl

"""

#import the necessary stuff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import InterpolatedUnivariateSpline
from ripser import ripser
from persim import plot_diagrams
import scipy.io.wavfile
from IPython.display import clear_output
from IPython.display import Audio
from MusicFeatures import *
import librosa
import librosa.display

def get_waveform(file_name, should_plot = False):
    #load in song and display it as waveform
    Fs, X = scipy.io.wavfile.read(file_name)
    X = X/(2.0**15) #in as 16 bit shorts, convert to float
    if(should_plot) :
        plt.figure()
        plt.plot(np.arange(len(X))/float(Fs), X)
        plt.xlabel("Time (secs)")
        plt.title("Song Name")
        plt.show()
    return Fs, X

def get_all_waveforms(filepaths):
    sampling_rates = []
    waveforms = []
    for path in filepaths:
        Fs, X = get_waveform(path)
        sampling_rates = sampling_rates.append(Fs)
        waveforms = waveforms.append(X)
    return sampling_rates, waveforms

#sliding window, assuming integer x, dim, Tau
def slidingWindowInt(x, dim, Tau, dT, duration = 3, mono = True):
    N = len(x)
    numWindows = int(np.floor((N-dim*Tau)/dT)) #number of windows
    if numWindows <= 0:
        print("Error: Tau too large")
        return np.zeros((duration, dim))
    X = np.zeros((numWindows, dim)) #2D array to store the windows
    idx = np.arange(N)
    for i in range(numWindows):
        #indices of the samples in window
        idxx = np.array(dT*i + Tau*np.arange(dim), dtype=np.int32)
        # This changes based on whether you have mono or stereo audio
        if(mono):
            X[i, :] = x[idxx]
        else:
            X[i, :] = x[idxx][:,0]
    return X


def compute_novfn(X, winSize, hopSize, plot_spectrogram = False, plot_novfn = False):
    #Compute the power spectrogram and audio novelty function
    (S, novFn) = getAudioNoveltyFn(X[:,0], Fs, winSize, hopSize)
        
    if plot_spectrogram:    
        plt.figure()
        plt.imshow(np.log(S.T), cmap = 'afmhot', aspect = 'auto')
        plt.title('Log-frequency power spectrogram')
        plt.show() 

    if plot_novfn:              
        plt.figure(figsize=(8, 4))
        #Plot the spectrogram again
        plt.subplot(211)
        plt.imshow(np.log(S.T), cmap = 'afmhot', aspect = 'auto')
        plt.ylabel('Frequency Bin')
        plt.title('Log-frequency power spectrogram')

        #Plot the audio novelty function
        plt.subplot(212)
        plt.plot(np.arange(len(novFn))*hopSize/float(Fs), novFn)
        plt.xlabel("Time (Seconds)")
        plt.ylabel('Audio Novelty')
        plt.xlim([0, len(novFn)*float(hopSize)/Fs])
        plt.show()

    return S,novFn

def compute_all_novfn(X_arr, winSizes, hopSizes):
    S_arr = []
    novFn_arr = []
    for i in len(X_arr):
        X = X_arr[i]
        winSize = winSizes[i]
        hopSize = hopSizes[i]
        S, novFn = compute_novfn(X, winSize, hopSize)
        S_arr = S_arr.append(S)
        novFn_arr = novFn_arr.append(novFn)
    return S_arr, novFn_arr


# Y is the data, dim is the MAX dimensional homology we want to compute
def compute_pd(Y, dim = 1, plot_points = False, plot_dgm = True):
    # Mean-center and normalize
    Y = Y - np.mean(Y, 1)[:, None]
    Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]
    PDs = ripser(Y, dim)['dgms']
    pca = PCA()
    Z = pca.fit_transform(Y)

    #print(pca.explained_variance_ratio_)

    # Plot point cloud and persistence diagram for song
    plt.figure(figsize=(8, 4))    
    if(plot_points):
        plt.subplot(121)
        plt.title("2D PCA")
        plt.scatter(Z[:, 0], Z[:, 1])
    if(plot_dgm):
        plt.subplot(122)
        plot_diagrams(PDs)
        plt.title("Persistence Diagram, dim = "+str(dim)+" tau = "+str(Tau) + "dT" + str(dT))
    if(plot_points or plot_dgm):
        plt.show()

    return PDs 

# This dimension actually means specifically the dimension of homology feature you want
def compute_all_pds(Y_arr, dim = 1):
    dgms = []
    for Y in Y_arr:
        PDs = compute_pd(Y, dim, plot_points = False, plot_dgm = False)
        dgms = dgms.append(PDs[dim])
    return dgms

# Compute and visualize clusters given list of diagrams
def compute_clusters(famemonster_dgms, artpop_dgms, chromatica_dgms, plot_clusters = True):
    all_dgms = np.concatenate([famemonster_dgms, artpop_dgms, chromatica_dgms])
    num_dgms = len(all_dgms)
    
    famemonster_length = len(famemonster_dgms)
    artpop_length = len(artpop_dgms)
    chromatica_length = len(chromatica_dgms)

    # Compute the distance matrix
    distance_matrix = np.empty([num_dgms,num_dgms])
    for i in range(num_dgms):
        for j in range(num_dgms):
            if i == j:
                distance_matrix[i][i] = 0
            else:
                distance_matrix[i][j] = persim.bottleneck(all_dgms[i], all_dgms[j])
    
    # Compute the MDS embedding
    embedding = sklearn.manifold.MDS(n_components = 2, random_state = 0, dissimilarity = 'precomputed')        
    coords = embedding.fit_transform(distance_matrix)
    x = coords[:,0]
    y = coords[:,1]

    # Album labels for the songs
    album = np.concatenate(np.full((len(famemonster_dgms),1), 0), np.full((len(artpop_dgms),1), 1). np.full((len(chromatica_dgms),1), 2))
    
    # Do kmeans clustering and get labels
    kmeans = sklearn.cluster.KMeans(n_clusters = 3).fit(coords)
    cluster_labels = kmeans.labels_
    print("I should have "+str(num_dgms)+"labels and I really have "+str(len(cluster_labels)))
    #plt.scatter(x,y,c=cluster_labels)
    
    # This is not a particularly elegant solution, but the simplest way I could get it to plot both different colors and different markers
    x_famemonster = x[:famemonster_length]
    y_famemonster = y[:famemonster_length]
    labels_famemonster = cluster_labels[:famemonster_length]
    
    x_artpop = x[famemonster_length:famemonster_length+artpop_length]
    y_artpop = y[famemonster_length:famemonster_length+artpop_length]
    labels_artpop = cluster_labels[famemonster_length:famemonster_length+artpop_length]
    x_chromatica = x[famemonster_length + artpop_length:famemonster_length+artpop_length + chromatica_length]
    y_chromatica = y[famemonster_length + artpop_length:famemonster_length+artpop_length + chromatica_length]
    labels_third_album = cluster_labels[famemonster_length + artpop_length:famemonster_length+artpop_length + chromatica_length]
    
    plt.scatter(x_famemonster, y_famemonster, c = labels_famemonster, marker = 'o')
    plt.scatter(x_artpop, y_artpop, c = labels_artpop, marker = 'x')
    plt.scatter(x_chromatica, y_chromatica, c = labels_third_album, marker = '+')


''' 
Basic Workflow:

For any given album...

1. Compute all waveforms and sampling rates
2. Compute all window sizes and hop sizes
3. Compute novelty functions (or chroma features, or etc) 
4. Compute persistence diagrams

Repeat for each album

5. Concatenate all lists of persistence diagrams, compute the clusters
'''

# Step 1
famemonster_filepaths = []
artpop_filepaths = []
chromatica_filepaths = []

famemonster_sample_rates, famemonster_waveforms = get_all_waveforms(famemonster_filepaths)
artpop_sample_rates, artpop_waveforms = get_all_waveforms(artpop_filepaths)
chromatica_sample_rates, chromatica_waveforms = get_all_waveforms(chromatica_filepaths)


# Step 2

# In beats per minute
# Might not have all the songs for famemonster
famemonster_tempos = [117,98,119,144,120,122,99,96]
artpop_tempos = [129,121,110,113,136,138,97,117,127,124,116,101,128,133,139]
chromatica_tempos = [75,123,117,123,117,117,75,117,121,120,117,123,121,122,123,117]



S, novFn = compute_novfn(X,winSize, hopSize)


# Step 3
# Can replace with whatever you want, novelty functions or chroma features or whatever

winSize = 512
hopSize = 256
famemonster_s, famemonster_novfns = compute_all_novfn(famemonster_waveforms, famemonster_winSizes, famemonster_hopSizes)
artpop_s, artpop_novfns = compute_all_novfn(artpop_waveforms, artpop_winSizes, artpop_hopSizes)
chromatica_s, chromatica_novfns = compute_all_novfn(chromatica_waveforms, chromatica_winSizes, chromatica_hopSizes)

# Step 4

# Y is the sliding window embedding
famemonster_pds = compute_all_pds(famemonster_novfns)
artpop_pds = compute_all_pds(artpop_novfns)
chromatica_pds = compute_all_pds(chromatica_novfns)


# Step 5

compute_clusters(famemonster_pds, artpop_pds, chromatica_pds)


#Take the first 3 seconds of the novelty function
fac = int(Fs/hopSize)
novFn = novFn[fac*4:fac*20]
# Chromatica I is at tempo of 66
# So window size should be about a second

#Make sure the window size is half of a second, noting that
#the audio novelty function has been downsampled by a "hopSize" factor
dim = 20
# Need Fs instead of Fs/2 because of the tempo thing - can implement a more general thing here later
Tau = (Fs)/(float(hopSize)*dim)
dT = 1

Y = slidingWindowInt(novFn, dim, Tau, dT)
print("Y.shape = ", Y.shape)

print("Sample rate"+str(Fs))
chroma = librosa.feature.chroma_stft(X[:,0], Fs)


print("Chroma shape: "+str(chroma.shape))
dim = 5
Tau = 1
dT = 3
Y_chroma = slidingWindowInt(chroma[3,0:200].transpose(), dim, Tau, dT)
print("Y_chroma.shape= ", Y_chroma.shape)
compute_pd(Y_chroma)










''' 
Random Old Code 


#dim*Tau here spans 1/2 second since Fs is the sample rate
Original settings 
dim = round(Fs/200)
Tau = 100
dT = Fs/100     

#dim = round(Fs/200)
dT = Fs/100     
tau_vals = [100]


for tau in tau_vals:
    dim = round(Fs/(2*tau))
    compute_pd(Fs, X, dim, tau, dT)


dim_arr = [5]
dt_arr = [2,5, 10]
for dim in dim_arr:     
    for dT in dt_arr: 
        Tau = (Fs)/(float(hopSize)*dim)

        Y = slidingWindowInt(novFn, dim, Tau, dT)
        print("Y.shape = ", Y.shape)
        print("dim: "+str(dim)+" dT: "+str(dT))
        compute_pd(Y)

Just print out the chromagram
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
'''