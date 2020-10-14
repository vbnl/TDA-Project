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

#sliding window, assuming integer x, dim, Tau
def slidingWindowInt(x, dim, Tau, dT, duration = 3):
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
        X[i, :] = x[idxx][:,0]
        #X[i, :] = x[idxx]
    return X

def compute_pd(Fs, X, dim, Tau, dT):
    # This version only looks at the first 3 seconds
    #Y = slidingWindowInt(X[0:Fs*3], dim, Tau, dT)
    Y = slidingWindowInt(X[0:Fs*3], dim, Tau, dT)
    
    print("Y.shape = ", Y.shape)
    #Mean-center and normalize
    Y = Y - np.mean(Y, 1)[:, None]
    Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]
    PDs = ripser(Y, maxdim=1)['dgms']
    pca = PCA()
    Z = pca.fit_transform(Y)

    #plot point cloud and persistence diagram for song
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.title("2D PCA")
    plt.scatter(Z[:, 0], Z[:, 1])
    plt.subplot(122)
    plot_diagrams(PDs)
    plt.title("Persistence Diagram, dim = "+str(dim)+" tau = "+str(Tau) + "dT" + str(dT))
    plt.show()

    #Compute the power spectrogram and audio novelty function
    winSize = 512
    hopSize = 256
    plt.figure()
    (S, novFn) = getAudioNoveltyFn(X[:,0], Fs, winSize, hopSize)
    plt.imshow(np.log(S.T), cmap = 'afmhot', aspect = 'auto')
    plt.title('Log-frequency power spectrogram')
    plt.show()

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

    (S, novFn) = getAudioNoveltyFn(X, Fs, winSize, hopSize)

    #Take the first 3 seconds of the novelty function
    fac = int(Fs/hopSize)
    novFn = novFn[fac*4:fac*7]

    #Make sure the window size is half of a second, noting that
    #the audio novelty function has been downsampled by a "hopSize" factor
    dim = 20
    Tau = (Fs/2)/(float(hopSize)*dim)
    dT = 1
    Y = getSlidingWindowInteger(novFn, dim, Tau, dT)
    print("Y.shape = ", Y.shape)
    #Mean-center and normalize
    Y = Y - np.mean(Y, 1)[:, None]
    Y = Y/np.sqrt(np.sum(Y**2, 1))[:, None]

    PDs = ripser(Y, maxdim=1)['dgms']
    pca = PCA()
    Z = pca.fit_transform(Y)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("2D PCA")
    plt.scatter(Z[:, 0], Z[:, 1])
    plt.subplot(122)
    plot_diagrams(PDs)
    plt.title("Persistence Diagram")
    plt.show()

Fs, X = get_waveform("01 Chromatica I.wav")

#dim*Tau here spans 1/2 second since Fs is the sample rate
''' Original settings 
dim = round(Fs/200)
Tau = 100
dT = Fs/100     
'''

#dim = round(Fs/200)
dT = Fs/100     
tau_vals = [100]


for tau in tau_vals:
    dim = round(Fs/(2*tau))
    compute_pd(Fs, X, dim, tau, dT)
