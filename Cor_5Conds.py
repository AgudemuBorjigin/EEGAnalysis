#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:20:42 2018

@author: baoagudemu1
"""

from anlffr.helper import biosemi2mne as bs
from anlffr.preproc import find_blinks #
from mne.preprocessing.ssp import compute_proj_epochs #
from mne.time_frequency import tfr_multitaper # calculating inter-trial coherence (phase-locking)
import mne
import numpy as np # support for large, multi-dimensional arrays and metrices
import pylab as pl
import os ### 
import fnmatch ###


# Adding Files and locations
froot = '/Users/baoagudemu1/Desktop/2018Spring/Lab/EEG-Python'
subjlist = ['S142',]

for subj in subjlist:
    fpath = froot + '/' + subj + '/'
    print 'Running subject', subj
    
    # load data and read event channels
    rawlist = []
    evelist = []
    # extracting bdf filenames into bdfs
    bdfs = fnmatch.filter(os.listdir(fpath), subj + '_ITD*.bdf') 
    if len(bdfs) >= 1:
        for k, bdf in enumerate(bdfs):
            rawtemp, evestemp = bs.importbdf(fpath + bdf)
            rawlist += [rawtemp,]
            evelist += [evestemp,]
    else:
        RuntimeError("No bdf files found!")
    
    # the returned eves has three columns: sample number, value of onset, value of offset.     
    raw, eves = mne.concatenate_raws(rawlist, events_list = evelist)
    # Things to try if data are too noisy
    # raw.info['bads'] += ['A7', 'A6', 'A24'] # if A7, A6, A24 have been rejected a lot, they can be added manually
    # raw.set_eeg_reference(ref_channels=['Average']) # referencing to the average of all channels, do this if channels are too noisy  
    
    eves2 = eves.copy()
    # the returned eves has three columns: sample number, value of onset, value of offset.
    # the eves are in 16 bits originally,
    # so this operation only looks at the lower 8 bits. Higher 8 bits are always 
    # high, representing button box condition. If interested in what's happening at button press, do np.floor(eves2[:,1]/256)
    eves2[:, 1] = np.mod(eves2[:, 1], 256) 
    eves2[:, 2] = np.mod(eves2[:, 2], 256)
    
    raw.filter(l_freq = 0.5, h_freq=50) # if channels are noisy, adjust the filter parameters
    # raw.plot(events=eves2)
    
    # SSP for blinks
    removeBlinks = True
    if removeBlinks:
        blinks = find_blinks(raw, ch_name = ['A1',], l_trans_bandwidth=0.5,
                             l_freq=1.0)
        # raw.plot(events=blinks) shows the lines at eye blinks
        
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
                                          n_mag=0, n_eeg=1,
                                          verbose='DEBUG') # greater n_eeg removes more nueral data
        raw.add_proj(blink_projs) # raw.del_proj()
        # raw.plot_projs_topomap() shows the 4 max PCAs (eigenvectors)
        
        # if channels are too noisy, play with n_eeg, if the second variance is acting more than
        # the first, that means the channels are contaminated not just by the eye blinks, but also
        # from other sources, using raw.plot(events = blinks, show_options = True)
        # raw.plot(events = blinks, show_options = True) could show the options for applying different projections
    
    # Average evoked response across conditions
    epochs = mne.Epochs(raw, eves2, [1, 2, 3, 4, 5, 6, 7, 8], tmin = -0.5, proj = True, tmax = 2.5, 
                        baseline = (-0.5, 0), reject = dict(eeg=100e-6)) # change the channels as needed
    # evoked = epochs.average() # always start with looking at evoked (averaged) response, 
    # and see which channels are bad by using and evoked.plot(picks=[30, 31]) and evoked.plot_topomap(times=[1.2])
    # add those channels to the list manually 
    # by raw.info['bads'] += ['A7', 'A6', 'A24'] if necessary
    
    #epochs.save(fpath+'/'+'no_blinks_epo.fif', split_size='2GB') # saving epochs into .fif format
    
    freqs = np.arange(5., 100., 2.)
    n_cycles = freqs/4. # time resolution is 0.25 s (1 s has "freq" number of cycles)
    time_bandwidth = 2.0 # number of taper = time_bandwidth product - 1 
    # usually n_cycles and time_bandwidth are fixed, which determines the frequency resolution 
    power, itc = tfr_multitaper(epochs, freqs = freqs,  n_cycles = n_cycles,
                   time_bandwidth = time_bandwidth, return_itc = True, n_jobs = 4)
    
    #np.savez(fpath+'/'+'itc_power', itc = itc.data, power = power.data) # saves multiple arrays, saved in .npz format
    # itc = itc seems to save the attributes of the itc instead of the real data
    # npzFile = np.load(fpath+'/'+'itc_power.npz'), npzFiles.files, npzFile['itc']
    # np.save(fpath+'/'+'itc_power', itc) # only one array, saved in .npy format
    
    itc_copy = itc.copy()
    # itc_copy.plot_topo(baseline = (-0.5, 0), mode = 'zscore') 
    itc_data = itc_copy.data
    # averaging across channels
    itc_data_mean = itc_data[[0, 1, 4, 8, 12, 21, 25, 26, 28, 29, 30, 31], :, :].mean(axis = 0) # these indexes should be changed if bad channels were added manually 
    #itc_data_mean = itc_data[[2, 3, 4, 5, 6, 7, 18, 20, 21, 22, 24, 25, 26, 27, 30, 31], :, :].mean(axis = 0)
    
    pl.imshow(itc_data_mean, interpolation='bicubic', aspect='auto', 
              origin='lower', cmap='RdBu_r') # the frequency axis doesn't seem to be right
    pl.figure()
    t = epochs.times
#   pl.plot(t, itc_data_mean[0:8, :].mean(axis=0))
    freqSub = np.where(freqs<22)
    itc_ave_below22Hz = itc_data_mean[freqSub[0], :].mean(axis=0)
    index = np.where(t>0)
    index1 = index[0]
    index1 = index1[0]
    index = np.where(t<0.2)
    index2 = index[0]
    index2 = index2[-1]
    
    noiseFloor = itc_ave_below22Hz[0:index1].mean(axis=0)
    itc_ave_below22Hz = itc_ave_below22Hz - noiseFloor
    
    firstPeakAmp = np.max(itc_ave_below22Hz[index1:index2])
    itc_norm_below22Hz = itc_ave_below22Hz/firstPeakAmp
    np.savez(froot+'/'+'itcs'+ '/'+'itc_avg'+'_'+subj, itc = itc_norm_below22Hz, t = t)
    pl.plot(t, itc_norm_below22Hz)
    pl.xlabel('Time (s)', fontsize=14)
    pl.ylabel('Normalized response', fontsize=14)
    pl.title('Phase locking: average across conditions', fontsize=14)
    ax = pl.gca()
    ax.tick_params(labelsize=14)
    pl.show()
    
#    pow_copy = power.copy()
#    pow_copy.plot_topo(baseline=(-0.5, 0), mode='zlogratio')
#    pow_copy.plot_topo(baseline=(-0.5, 0), mode='logratio')
#    pow_copy.plot_topo(baseline=(-0.5, 0), mode='mean')
    
#    # my code
#    itc_copy.plot(picks = [31,], baseline = (-0.5, 0), mode = 'zscore', vmin = 0, vmax = 0.4) # itc has 32 channels in the beginning
#    # itc_copy.plot_topo(baseline=(-0.5, 0), mode='zscore') # plot_topo produces topo of tf plots, zscore is obtained by subtracting the mean of baseline from data points
#    # and dividing them by the std of baseline
#    itc_chann = itc_copy.pick_channels([u'A31',]) # channel names are usually in the form of u'A**'
#    itc_data = itc_chann.data[0, np.where(freqs<20),]
#    itc_data_ave = np.mean(itc_data, axis = 1)
#    itc_data_ave_nor = itc_data_ave/np.max(itc_data_ave[0,])
#    pl.figure()
#    t = epochs.times
#    pl.plot(t, itc_data_ave_nor[0,]) # 3-D data
#    pl.xlabel('Time (s)', fontsize=14)
#    pl.ylabel('Normalized response', fontsize=14)
#    pl.title('Phase locking', fontsize=14)
#    ax = pl.gca()
#    ax.tick_params(labelsize=14)
#    pl.show()
    

    
    # Single condition visualization
    #condnames = ['20', '140', '260', '380', '500']
    condlist = [1, 2, 3, 4] # change as needed
    condnames = ['20', '60', '180', '540']
    evokeds = []
    powers = []
    itcs = []
    itcs_data = []
    powers_data = []
    for k, name in enumerate(condnames):
        cond = condlist[k]
        print 'Doing condition', cond
        epochs_1cond = mne.Epochs(raw, eves2, [cond, cond+4], tmin = -0.5, proj = True, tmax = 2.5, 
                        baseline = (-0.5, 0), reject = dict(eeg=100e-6)) # change the channels as needed
#        epochs_1cond = epochs[cond].average() # it extracts the single events instead of the conditions as a whole
#        evoked_1cond = epochs_1cond.average()
#        evoked_1cond.plot(picks=[30, 31])
#        evoked_1cond.plot_topomap(times=[1.2])
#        evoked_1cond.pick_channels()
#        evokeds += [epochs_1cond.average(),] # plus seems to mean concatenation
        #epochs_1cond.save(fpath+'/'+'no_blinks_epo_cond'+str(cond)+'.fif', split_size='2GB')
        power_1cond, itc_1cond = tfr_multitaper(epochs_1cond, freqs = freqs,  n_cycles = n_cycles,
                   time_bandwidth = time_bandwidth, return_itc = True, n_jobs = 4)
        powers += [power_1cond,]
        itcs += [itc_1cond,]
    
    for k, name in enumerate(condnames):
        itcs_data += [itcs[k].data,]
        powers_data += [powers[k].data,]
    
    #np.savez(fpath+'/'+'conds_itc_power', itcs = itcs_data, powers = powers_data)
    
    cond = 4
    itc_copy_cond = itcs[cond-1].copy()
    itc_data_cond = itc_copy_cond.data
    # averaging across channels
    itc_data_mean_cond = itc_data_cond[[0, 1, 4, 8, 12, 21, 25, 26, 28, 29, 30, 31], :, :].mean(axis = 0) 
#    pl.imshow(itc_data_mean_cond, interpolation='bicubic', aspect='auto', 
#              vmin=0.1, vmax=0.4, origin='lower', cmap='RdBu_r')
    pl.imshow(itc_data_mean_cond, interpolation='bicubic', aspect='auto', 
              origin='lower', cmap='RdBu_r')
    pl.figure()
    t = epochs.times
    freqSub = np.where(freqs<22)
    itc_ave_below22Hz_cond = itc_data_mean_cond[freqSub[0], :].mean(axis=0)
    index = np.where(t>0)
    index1 = index[0]
    index1 = index1[0]
    index = np.where(t<0.2)
    index2 = index[0]
    index2 = index2[-1]
    
    noiseFloor_cond = itc_ave_below22Hz_cond[0:index1].mean(axis=0)
    itc_ave_below22Hz_cond = itc_ave_below22Hz_cond - noiseFloor_cond
    
    firstPeakAmp_cond = np.max(itc_ave_below22Hz_cond[index1:index2])
    itc_norm_below22Hz_cond = itc_ave_below22Hz_cond/firstPeakAmp_cond
    np.savez(froot+'/'+'itcs'+ '/' +'itc_avg'+str(cond)+'_'+subj, itc = itc_norm_below22Hz_cond, t = t)
    pl.plot(t, itc_norm_below22Hz_cond)
    pl.xlabel('Time (s)', fontsize=14)
    pl.ylabel('Normalized response', fontsize=14)
    pl.title('Phase locking: condtion '+str(cond), fontsize=14)
    ax = pl.gca()
    ax.tick_params(labelsize=14)
    pl.show()
        
    
    
#    # plot data
#    pl.figure()
##    for k in condlist:
#    for k in [5]:
#        evoked = evokeds[k-1]
#        x = evoked.data * 1e6 # microV 
#        t = evoked.times *1e3 - 1.6 ### Adjust for delay and use milliseconds
#        # y = x.mean(axis = 0) # this step is actually not recommended for 
#        # cortical response, since polarity matters in this case
#        pl.plot(t, x[31,:], linewidth=2)
#    pl.xlabel('Time (ms)', fontsize=14)
#    pl.ylabel('Cortical response (uV)', fontsize=14)
#    pl.title('ITD', fontsize=14)
#    #pl.xlim((0., 10.))
#    #pl.ylim((-1.0, 1.5))
#    ax = pl.gca()
#    ax.tick_params(labelsize=14)
##    pl.legend(['0.1', '0.3368', '0.5736', '0.8104', '1.0472', '0'])
#    pl.legend(['20', '140', '260', '380', '500'])
#    pl.show()
#    # simple plotting: evokeds[5].plot(picks = [30, 31])
#    # pl.close('all')
#    
#    pl.figure()
##    for k in condlist:
#    for k in [5]:
#        evoked = evokeds[k-1]
#        x = evoked.data * 1e6 # microV
#        t = evoked.times *1e3 - 1.6
##        fs = raw.info['sfreq']
##        tn = len(t)
##        yf = np.abs(fft.fft(y))
##        xf = np.linspace(0, fs/2, tn/2)
##        pl.plot(xf, yf[:tn//2], linewidth=2)
##        The following commands takes 2 or 3 dimentional data, and takes each dimention 
##        as event or trial        
##        params = dict(Fs=raw.info['sfreq'], fpass=[1, 100], tapers=[1, 1], noisefloortype=0)
##        s, n, f = spectral.mtspec(x,params)
#        f = fft.fftfreq(x[31,:].shape[0])*evoked.info['sfreq']
#        s = fft.fft(x[31,:])
#        pl.plot(f, np.abs(s), linewidth=2)
#        
#    pl.xlabel('Frequency (Hz)', fontsize=14)
#    pl.ylabel('Amplitude', fontsize=14)
#    pl.title('Frequency response', fontsize=14)
#    ax = pl.gca()
#    ax.tick_params(labelsize=14)
##   pl.legend(['0.1', '0.3368', '0.5736', '0.8104', '1.0472', '0'])
#    pl.show()
    
    


    
        
        

    
        
    

    
    
        
            
    
    