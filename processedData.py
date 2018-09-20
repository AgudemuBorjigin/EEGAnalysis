#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:20:40 2018

@author: baoagudemu1
"""
import mne
import numpy as np # support for large, multi-dimensional arrays and metrices
import pylab as pl
import os ### 
import fnmatch ###

# Adding Files and locations
#froot = '/Users/baoagudemu1/Desktop/2018Spring/Lab/EEG-Python'
froot = '/media/agudemu/Storage/Data/EEG/ITD/itc_avg_auditory'
subjlist = ['itcs_avg_channels_below40Hz',]

for subj in subjlist:
    fpath = froot + '/' + subj + '/'
    print 'Running subject', subj
    
    epochs_condlist = []
    epochs_avglist = []
    fifs_avg = fnmatch.filter(os.listdir(fpath), 'no_blinks_epo*.fif') # note the file naming format
    fifs_cond = fnmatch.filter(os.listdir(fpath), 'no_blinks_epo_cond*.fif') #note the file naming format,
    # what if no_blinks_epo_cond*.fif has to be stored in 2 files?
    
    if len(fifs_cond) >= 1 or len(fifs_avg) >=1:
        for k, fif_avg in enumerate(fifs_avg):
            epochs_avg_temp = mne.read_epochs(fpath + fif_avg)
            epochs_avglist += [epochs_avg_temp,] # how to put them together into one epochs?
        for k, fif_cond in enumerate(fifs_cond): 
            epochs_cond_temp = mne.read_epochs(fpath + fif_cond)
            epochs_condlist += [epochs_cond_temp,]
    else:
        RuntimeError("No fif files found!")
    
    itc_avgList = []
    itc_avg_npzs = fnmatch.filter(os.listdir(fpath), 'itc_avg_S*.npz')
    
    if len(itc_avg_npzs) >= 1:
        for k, itc_avg_npz in enumerate(itc_avg_npzs):
            itc_avg_temp = np.load(fpath + itc_avg_npz)
            itc_avgList += [itc_avg_temp['itc'],]     
    else:
        RuntimeError("No npz files found!")    
    t_temp =  np.load(fpath + itc_avg_npz)   
    t = t_temp['t']
    itc_avg_mean = sum(itc_avgList)/(k+1)
    
    ################
    itc_avg1List = []
    
    itc_avg1_npzs = fnmatch.filter(os.listdir(fpath), 'itc_avg1_S*.npz')
    
    if len(itc_avg1_npzs) >= 1:
        for k, itc_avg_npz in enumerate(itc_avg1_npzs):
            itc_avg_temp = np.load(fpath + itc_avg_npz)
            itc_avg1List += [itc_avg_temp['itc'],]     
    else:
        RuntimeError("No npz files found!")    
    itc_avg1_mean = sum(itc_avg1List)/(k+1)
    
    ################
    itc_avg2List = []
    
    itc_avg2_npzs = fnmatch.filter(os.listdir(fpath), 'itc_avg2_S*.npz')
    
    if len(itc_avg2_npzs) >= 1:
        for k, itc_avg_npz in enumerate(itc_avg2_npzs):
            itc_avg_temp = np.load(fpath + itc_avg_npz)
            itc_avg2List += [itc_avg_temp['itc'],]     
    else:
        RuntimeError("No npz files found!")    
    itc_avg2_mean = sum(itc_avg2List)/(k+1)
    
    ################
    itc_avg3List = []
    
    itc_avg3_npzs = fnmatch.filter(os.listdir(fpath), 'itc_avg3_S*.npz')
    
    if len(itc_avg3_npzs) >= 1:
        for k, itc_avg_npz in enumerate(itc_avg3_npzs):
            itc_avg_temp = np.load(fpath + itc_avg_npz)
            itc_avg3List += [itc_avg_temp['itc'],]     
    else:
        RuntimeError("No npz files found!")    
    itc_avg3_mean = sum(itc_avg3List)/(k+1)
    
    ################
    itc_avg4List = []
    
    itc_avg4_npzs = fnmatch.filter(os.listdir(fpath), 'itc_avg4_S*.npz')
    
    if len(itc_avg4_npzs) >= 1:
        for k, itc_avg_npz in enumerate(itc_avg4_npzs):
            itc_avg_temp = np.load(fpath + itc_avg_npz)
            itc_avg4List += [itc_avg_temp['itc'],]     
    else:
        RuntimeError("No npz files found!")    
    itc_avg4_mean = sum(itc_avg4List)/(k+1)
    
    pl.figure()
    avg, = pl.plot(t, itc_avg_mean, label = 'Avg across conditions')
    avg1, = pl.plot(t, itc_avg1_mean, label = 'ITD = 20 us')
    avg2, = pl.plot(t, itc_avg2_mean, label = 'ITD = 60 us')
    avg3, = pl.plot(t, itc_avg3_mean, label = 'ITD = 180 us')
    avg4, = pl.plot(t, itc_avg4_mean, label = 'ITD = 540 us')
    pl.xlabel('Time (s)', fontsize=14)
    pl.ylabel('Normalized response', fontsize=14)
    pl.title('Phase locking: avg across subjects', fontsize=14)
    ax = pl.gca()
    ax.tick_params(labelsize=14)
    pl.legend(handles = [avg, avg1, avg2, avg3, avg4])
    pl.show()
        
    
    
        


        