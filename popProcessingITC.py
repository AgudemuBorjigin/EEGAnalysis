#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:20:40 2018

@author: baoagudemu1
"""
import mne
import numpy as np # support for large, multi-dimensional arrays and metrices
import pylab as pl
import os #
import fnmatch #
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio

# Adding Files and locations
froot = '/media/agudemu/Storage/Data/EEG/ITD/itcCz/'

itc_avgList = []
itc_avg_npzs = fnmatch.filter(os.listdir(froot), 'S*_itcavg.npz')

if len(itc_avg_npzs) >= 1:
    for k, itc_avg_npz in enumerate(itc_avg_npzs):
        itc_avg_temp = np.load(froot + itc_avg_npz)
        itc_avgList += [itc_avg_temp['itc_avg'],]     
else:
    RuntimeError("No npz files found!")    
t_temp =  np.load(froot + itc_avg_npz)   
t = t_temp['t']
itc_avg_mean = sum(itc_avgList)/(k+1)

################
itc_avg1List = []

itc_avg1_npzs = fnmatch.filter(os.listdir(froot), 'S*_itc20us.npz')

if len(itc_avg1_npzs) >= 1:
    for k, itc_avg_npz in enumerate(itc_avg1_npzs):
        itc_avg_temp = np.load(froot + itc_avg_npz)
        itc_avg1List += [itc_avg_temp['itc_avg'],]     
else:
    RuntimeError("No npz files found!")    
itc_avg1_mean = sum(itc_avg1List)/(k+1)

################
itc_avg2List = []

itc_avg2_npzs = fnmatch.filter(os.listdir(froot), 'S*_itc60us.npz')

if len(itc_avg2_npzs) >= 1:
    for k, itc_avg_npz in enumerate(itc_avg2_npzs):
        itc_avg_temp = np.load(froot + itc_avg_npz)
        itc_avg2List += [itc_avg_temp['itc_avg'],]     
else:
    RuntimeError("No npz files found!")    
itc_avg2_mean = sum(itc_avg2List)/(k+1)

################
itc_avg3List = []

itc_avg3_npzs = fnmatch.filter(os.listdir(froot), 'S*_itc180us.npz')

if len(itc_avg3_npzs) >= 1:
    for k, itc_avg_npz in enumerate(itc_avg3_npzs):
        itc_avg_temp = np.load(froot + itc_avg_npz)
        itc_avg3List += [itc_avg_temp['itc_avg'],]     
else:
    RuntimeError("No npz files found!")    
itc_avg3_mean = sum(itc_avg3List)/(k+1)

################
itc_avg4List = []

itc_avg4_npzs = fnmatch.filter(os.listdir(froot), 'S*_itc540us.npz')

if len(itc_avg4_npzs) >= 1:
    for k, itc_avg_npz in enumerate(itc_avg4_npzs):
        itc_avg_temp = np.load(froot + itc_avg_npz)
        itc_avg4List += [itc_avg_temp['itc_avg'],]     
else:
    RuntimeError("No npz files found!")    
itc_avg4_mean = sum(itc_avg4List)/(k+1)

def plot_spectrogram(itc_data_mean, t, freqs, cond):
    fontsize = 45
    fig, ax = pl.subplots()
    handle = ax.imshow(itc_data_mean, interpolation='bicubic', aspect='auto', 
              origin='lower', cmap= 'jet', vmin = 0, vmax = 0.38) 
    # ax.set_xticks(np.arange(t[0], t[-1]+0.5,0.5));
    ax.set_xticks(np.arange(0, len(t), len(t)/6));
    ax.set_xticklabels(['-0.5', '0', '0.5', '1', '1.5', '2.0', '2.5'], fontsize = fontsize)
    ax.set_yticks(np.arange(0, len(freqs), len(freqs)/5));
    ax.set_yticklabels(['1', '9', '18', '27', '36', '45'], fontsize = fontsize)
    ax.set_title('ITC: averaged across 42 subjects: ITD = '+str(cond), fontsize = fontsize)
    #ax.set_xlabel('Time [s]', fontsize = fontsize)
    ax.set_ylabel('Frequency [Hz]', fontsize = fontsize)
    pl.show()
    cbar = pl.colorbar(handle)
    cbar.ax.tick_params(labelsize = fontsize)

def noiseFloorEstimate(t, evoked):
    index = np.where(t>0)
    index1 = index[0]
    index1 = index1[0]
    index = np.where(t<0.3) # CHANGE AS NEEDED FOR DIFFERENT SUBJECTS, 0.3
    index2 = index[0]
    index2 = index2[-1]
    
    noiseFloor = evoked[0:index1].mean(axis=0)
    return noiseFloor, index1, index2
   
def itc_normalization(itc, freqs, t):
    freqSub = np.where(freqs<20)
        
    itc_ave = itc[freqSub[0], :].mean(axis=0)
    noiseFloor, index1, index2 = noiseFloorEstimate(t, itc_ave)
    itc_ave = itc_ave - noiseFloor
    
    firstPeakAmp = np.absolute(np.max(itc_ave[index1:index2])) 
    itc_norm = itc_ave/firstPeakAmp
    return itc_norm

freqs = np.arange(1., 50., 1.) # chaneg as needed
plot_spectrogram(itc_avg4_mean, t, freqs, '540 us')
itc_norm4 = itc_normalization(itc_avg4_mean, freqs, t)
itc_norm3 = itc_normalization(itc_avg3_mean, freqs, t)
itc_norm2 = itc_normalization(itc_avg2_mean, freqs, t)
itc_norm1 = itc_normalization(itc_avg1_mean, freqs, t)
itc_norm = itc_normalization(itc_avg_mean, freqs, t)

fontSize = 45
lineWidth = 3
#fig = pl.figure(figsize = (25, 9))
pl.figure()
avg, = pl.plot(t, itc_norm, label = 'Avg across conditions', linewidth = lineWidth)
avg1, = pl.plot(t, itc_norm1, label = 'ITD = 20 us', linewidth = lineWidth)
avg2, = pl.plot(t, itc_norm2, label = 'ITD = 60 us', linewidth = lineWidth)
avg3, = pl.plot(t, itc_norm3, label = 'ITD = 180 us', linewidth = lineWidth)
avg4, = pl.plot(t, itc_norm4, label = 'ITD = 540 us', linewidth = lineWidth)
#pl.xlabel('Time [s]', fontsize=fontSize)
pl.ylabel('Normalized ITC re: onset', fontsize=fontSize)
#pl.title('Phase locking strength: averaged across 42 NH subjects', fontsize=fontSize)
ax = pl.gca()
ax.tick_params(labelsize=fontSize)
hleg = pl.legend(handles = [avg, avg1, avg2, avg3, avg4], prop = {'size':25}, loc = 'best')
pl.show()
            
        


        