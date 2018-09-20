#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 12:32:03 2018

@author: agudemu
"""
from anlffr.helper import biosemi2mne as bs
from anlffr.preproc import find_blinks #
from mne.preprocessing.ssp import compute_proj_epochs #
from mne.time_frequency import tfr_multitaper # calculating inter-trial coherence (phase-locking)
import mne
import numpy as np # support for large, multi-dimensional arrays and metrices
import pylab as pl
import os # it assigns its path attribute to an os-specific path module
import fnmatch # unix filename pattern matching

subID = 'S149'

#####################################################################################################################################
itc_avg = np.load('/home/agudemu/Data/EEG/ITD'+'/'+'itcs_avg_channels'+ '/'+'itc_avg_'+subID + '.npz')
itc_data_mean = itc_avg['itc']
t = itc_avg['t']
freqs = itc_avg['freqs']

fig, ax = pl.subplots()
im = ax.imshow(itc_data_mean, interpolation='bicubic', aspect='auto', 
          origin='lower', cmap='RdBu_r', vmin = 0, vmax = 0.2) 
# ax.set_xticks(np.arange(t[0], t[-1]+0.5,0.5));
ax.set_xticks(np.arange(0, len(t), len(t)/6));
ax.set_xticklabels(['-0.5', '0', '0.5', '1', '1.5', '2.0', '2.5'])
ax.set_yticks(np.arange(0, len(freqs), len(freqs)/11));
ax.set_yticklabels(['1', '5', '9', '13', '17', '21', '25', '29', '33', '37', '41', '45', '49'])
ax.set_title('Inter-trial coherence averaged across channels: across conditions')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Frequency [Hz]')
pl.show()

pl.figure()
#   pl.plot(t, itc_data_mean[0:8, :].mean(axis=0))
freqSub = np.where(freqs<23) # CHANGE AS NEEDED FOR DIFFERENT SUBJECTS, refer to time-freq plot of itc
itc_ave_below22Hz = itc_data_mean[freqSub[0], :].mean(axis=0)
index = np.where(t>0)
index1 = index[0]
index1 = index1[0]
index = np.where(t<0.3) # CHANGE AS NEEDED FOR DIFFERENT SUBJECTS
index2 = index[0]
index2 = index2[-1]

noiseFloor = itc_ave_below22Hz[0:index1].mean(axis=0)
itc_ave_below22Hz = itc_ave_below22Hz - noiseFloor

firstPeakAmp = np.max(itc_ave_below22Hz[index1:index2])
itc_norm_below22Hz = itc_ave_below22Hz/firstPeakAmp

pl.plot(t, itc_norm_below22Hz)
pl.xlabel('Time (s)', fontsize=14)
pl.ylabel('Normalized response', fontsize=14)
pl.title('Phase locking: average across conditions', fontsize=14)
ax = pl.gca()
ax.tick_params(labelsize=14)
pl.show()

####################################################################################################################################
cond = 1
itc_avg_cond = np.load('/home/agudemu/Data/EEG/ITD'+'/'+'itcs_avg_channels'+ '/'+'itc_avg'+str(cond)+ subID +'.npz')
itc_data_mean_cond = itc_avg_cond['itc']

fig, ax = pl.subplots()
im = ax.imshow(itc_data_mean_cond, interpolation='bicubic', aspect='auto', 
          origin='lower', cmap='RdBu_r', vmin = 0, vmax = 0.38)
ax.set_xticks(np.arange(0, len(t), len(t)/6));
ax.set_xticklabels(['-0.5', '0', '0.5', '1', '1.5', '2.0', '2.5'])
ax.set_yticks(np.arange(0, len(freqs), len(freqs)/11));
ax.set_yticklabels(['1', '5', '9', '13', '17', '21', '25', '29', '33', '37', '41', '45', '49'])
ax.set_title('Inter-trial coherence averaged across channels: condition '+str(cond))
ax.set_xlabel('Time [s]')
ax.set_ylabel('Frequency [Hz]')
pl.show()

pl.figure()
freqSub = np.where(freqs<23) # CHANGE AS NEEDED FOR DIFFERENT SUBJECTS, refer to time-freq plot of itc
itc_ave_below22Hz_cond = itc_data_mean_cond[freqSub[0], :].mean(axis=0)
index = np.where(t>0)
index1 = index[0]
index1 = index1[0]
index = np.where(t<0.3) # CHANGE AS NEEDED FOR DIFFERENT SUBJECTS
index2 = index[0]
index2 = index2[-1]

noiseFloor_cond = itc_ave_below22Hz_cond[0:index1].mean(axis=0)
itc_ave_below22Hz_cond = itc_ave_below22Hz_cond - noiseFloor_cond

firstPeakAmp_cond = np.max(itc_ave_below22Hz_cond[index1:index2])
itc_norm_below22Hz_cond = itc_ave_below22Hz_cond/firstPeakAmp_cond
pl.plot(t, itc_norm_below22Hz_cond)
pl.xlabel('Time (s)', fontsize=14)
pl.ylabel('Normalized response', fontsize=14)
pl.title('Phase locking: condtion '+str(cond), fontsize=14)
ax = pl.gca()
ax.tick_params(labelsize=14)
pl.show()