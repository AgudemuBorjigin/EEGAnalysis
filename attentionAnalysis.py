#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:32:16 2018

@author: baoagudemu1
"""

from anlffr.helper import biosemi2mne as bs
from anlffr.preproc import find_blinks 
from mne.preprocessing.ssp import compute_proj_epochs 
from mne.time_frequency import tfr_multitaper # calculating inter-trial coherence (phase-locking)
import mne
import numpy as np # support for large, multi-dimensional arrays and metrices
import pylab as pl
import os 
import fnmatch 


# Adding Files and locations
#froot = '/Users/baoagudemu1/Desktop/2018Spring/Lab/EEG-Python/Atten'
froot = '/home/agudemu/Data/EEG/Atten'
subjlist = ['S011',]

for subj in subjlist:
    fpath = froot + '/' + subj + '/'
    print 'Running subject', subj
    
    # load data and read event channels
    rawlist = []
    evelist = []
    # extracting bdf filenames into bdfs
    bdfs = fnmatch.filter(os.listdir(fpath), subj + '_Atten*.bdf') 
    if len(bdfs) >= 1:
        for k, bdf in enumerate(bdfs):
            rawtemp, evestemp = bs.importbdf(fpath + bdf)
            rawlist += [rawtemp,]
            evelist += [evestemp,] # Are events extracted by the TRIGGERS?? channels?? thresholds??
    else:
        RuntimeError("No bdf files found!")
        
    raw, eves = mne.concatenate_raws(rawlist, events_list = evelist) # 
    # Things to try if data are too noisy
    # raw.info['bads'] += ['A7', 'A6', 'A24'] # if A7, A6, A24 have been rejected a lot, they can be added manually
    # raw.set_eeg_reference(ref_channels=['Average']) # referencing to the average of all channels, do this if channels are too noisy  
    
    eves2 = eves.copy()
    # the returned eves has three columns: sample number, value of onset, value of offset.
    # the eves are in 16 bits originally,
    # so this operation only looks at the lower 8 bits. Higher 8 bits are always 
    # high, representing button box condition. If interested in what's happening at button press, 
    # do np.floor(eves2[:,1]/256)
    eves2[:, 1] = np.mod(eves2[:, 1], 256) # second column
    eves2[:, 2] = np.mod(eves2[:, 2], 256) # third column
    
    raw.filter(l_freq = 0.5, h_freq=50) # if channels are noisy, adjust the filter parameters
    # raw.plot(events=eves2)
    
    # SSP for blinks
    removeBlinks = True
    if removeBlinks:
        blinks = find_blinks(raw, ch_name = ['A1',], l_trans_bandwidth=0.5,
                             l_freq=1.0)
        blink_eves2 = np.concatenate((eves2, blinks), axis = 0)
        # raw.plot(events=blinks) shows the lines at eye blinks
        # blink and eves2 triggers can be combined using np.concatenate((eves2, blinks), axis = 0)
        
        # the trigger for blinks can be chosen to be starting from 1000, just to make sure it doesn't collide with the triggers for conditions
        epochs_blinks = mne.Epochs(raw, blinks, 998, tmin = -0.5, tmax = 0.5, 
                                   proj = False, baseline = (-0.5, 0), 
                                   reject=dict(eeg=500e-6)) 
        # PCA is only applied to the epochs around eye blinks. Since the eye blinks are 
        # contributing the most to the variance within this chunk of window, 
        # the first PCA (first eigenvector) is going to be due to the eye blink 
        # for sure and removed. If the PCA was performed on the whole samples, we wouldn't
        # know which vector is going to be linked to the eye blink
        # for "n_eeg", it's recommended to remove only the biggest projection, which is eye blinks in this case
    
        blink_projs = compute_proj_epochs(epochs_blinks, n_grad=0,
                                          n_mag=0, n_eeg=4,
                                          verbose='DEBUG') # greater n_eeg removes more nueral data
        # raw.plot_projs_topomap() shows the 4 max PCAs (eigenvectors)
        
        # if channels are too noisy, play with n_eeg, if the second variance is acting more than
        # the first, that means the channels are contaminated not just by the eye blinks, but also
        # from other sources, using raw.plot(events = blinks, show_options = True)
        # raw.plot(events = blinks, show_options = True) could show the options for applying different projections
        raw.add_proj(blink_projs) # opposite: raw.del_proj()
        
        # MANUALLY SELECT PROJECTIONS BY PLOTTING raw.plot_projs_topomap
        # REMOVE EXTRA PROJS USING raw.del_proj -- Remember index starts at 0
    
    # Average evoked response across short-stream conditions
    epochs_short = mne.Epochs(raw, eves2, [3, 4, 7, 8], tmin = -0.5, proj = True, tmax = 4.2, 
                        baseline = (-0.5, 0.), reject = dict(eeg=150e-6)) # change the channels as needed
    evoked_shortStream = epochs_short.average() 
    # evoeked_shortStream.plot_topo()
    # evoked = epochs.average() # always start with looking at evoked (averaged) response, 
    # and see which channels are bad by using and evoked.plot(picks=[30, 31]) and evoked.plot_topomap(times=[1.2]) for instance
    # add those channels to the bad-channel list manually 
    # by raw.info['bads'] += ['A7', 'A6', 'A24'] if necessary
    # Note: evoked.plot_topo and evoked.plot_topomap are different
    
    # Average evoked response across long-stream conditions
    epochs_long = mne.Epochs(raw, eves2, [1, 2, 5, 6], tmin = -0.5, proj = True, tmax = 4.2, 
                        baseline = (-0.5, 0), reject = dict(eeg=150e-6))
    evoked_longStream = epochs_long.average()

    # Average visual evoked response
    epochs_cue = mne.Epochs(raw, eves2, [9, 10, 11, 12, 13, 14, 15, 16],
                            tmin = -0.5, proj = True, tmax = 1.0, 
                            baseline = (-0.5, 0), reject = dict(eeg=150e-6))
    evoked_cue = epochs_cue.average()
    
    
    # computation of inter-trial-coherence (itc)
    freqs = np.arange(5., 100., 2.)
    n_cycles = freqs/4. # time resolution is 0.25 s (1 s has "freq" number of cycles)
    time_bandwidth = 2.0 # number of taper = time_bandwidth product - 1 
    # usually n_cycles and time_bandwidth are fixed, which determines the frequency resolution 
    # short-stream condition
    power_short, itc_short = tfr_multitaper(epochs_short, freqs = freqs,  n_cycles = n_cycles,
                   time_bandwidth = time_bandwidth, return_itc = True, n_jobs = 4)
    itc_copy_short = itc_short.copy()
    itc_data_short = itc_copy_short.data
    # averaging across channels
    itc_data_short_mean = itc_data_short.mean(axis = 0) # these indexes should be changed 
    # if bad channels were added manually
    pl.imshow(itc_data_short_mean, interpolation='bicubic', aspect='auto', 
              origin='lower', cmap='RdBu_r') # the frequency axis doesn't seem to be right
    
    pl.figure()
    t = epochs_short.times
    freqSub = np.where(freqs<22)
    itc_ave_below22Hz_short = itc_data_short_mean[freqSub[0], :].mean(axis=0)
    index = np.where(t>0)
    index1 = index[0]
    index1 = index1[0]
    index = np.where(t<0.2) # CHNAGE AS NEEDED FOR DIFFERENT SUBJECTS
    index2 = index[0]
    index2 = index2[-1]
    
    noiseFloor = itc_ave_below22Hz_short[0:index1].mean(axis=0)
    itc_ave_below22Hz_short = itc_ave_below22Hz_short - noiseFloor
    
    firstPeakAmp = np.absolute(np.max(itc_ave_below22Hz_short[index1:index2])) # ABSOLUTE? POLARITY MATTERS
    itc_norm_below22Hz = itc_ave_below22Hz_short/firstPeakAmp
    np.savez(froot+'/'+'itcs'+ '/'+'itc_avg'+'_'+subj, itc = itc_norm_below22Hz, t = t)
    pl.plot(t, itc_norm_below22Hz)
    pl.xlabel('Time (s)', fontsize=14)
    pl.ylabel('Normalized response', fontsize=14)
    pl.title('Phase locking: average across short conditions', fontsize=14)
    ax = pl.gca()
    ax.tick_params(labelsize=14)
    pl.show()