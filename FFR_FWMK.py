#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 16:25:08 2019

@author: agudemu
"""

from anlffr.helper import biosemi2mne as bs
from anlffr.spectral import mtplv
import mne
import numpy as np # support for large, multi-dimensional arrays and metrices
import pylab as pl
import os # it assigns its path attribute to an os-specific path module
import fnmatch # unix filename pattern matching
from scipy.signal import butter, filtfilt, hilbert
from mne.time_frequency import tfr_multitaper
OS = 'Ubuntu'

if OS == 'Ubuntu':
    froot = '/media/agudemu/Storage/Data/EEG/FM/FWMK'
else:
    froot = '/Users/baoagudemu1/Desktop/Lab/EEG-Python/FM'

subjlist = ['S116']     

def processing(fpath, bdf, eid, fs): # eid = [id_475_pos, id_475_neg, id_500_pos, id_500_neg, id_525_pos, id_525_neg,]
    rawtemp, evestemp = bs.importbdf(fpath + bdf)
    raw = rawtemp
    eves2 = evestemp.copy()
    eves2[:, 1] = np.mod(eves2[:, 1], 256) 
    eves2[:, 2] = np.mod(eves2[:, 2], 256)
    raw.filter(l_freq = 400, h_freq=1300) # adjust this range 
    evoked_pos = []
    evoked_neg = []
    
    numpos = []
    numneg = []
    
    flags = [False, False]
    if eid[0] in eves2[:,2]:
        evoked_pos, numpos = evoked(raw, eves2, eid[0])
        subtraction = offsetSub(evoked_pos, fs)
        evoked_pos = evoked_pos - subtraction
        evoked_pos = evoked_pos[int(0.0266*fs):int((0.0266+0.395)*fs)] # starting from time 0 to the end of masked stimulus   
        flags[0] = True;
        length = len(evoked_pos)
    if eid[1] in eves2[:,2]:
        evoked_neg, numneg = evoked(raw, eves2, eid[1])
        subtraction = offsetSub(evoked_neg, fs)
        evoked_neg = evoked_neg - subtraction 
        evoked_neg = evoked_neg[int(0.0266*fs):int((0.0266+0.395)*fs)]   
        flags[1] = True;
        length = len(evoked_neg)
    
    return evoked_pos, evoked_neg, numpos, numneg, length, flags

def offsetSub(evoked, fs):
    offset = evoked[int(0.525*fs):int(0.555*fs)+1] # add 0.025 s
    subtraction = np.concatenate((np.zeros(int(0.275*fs)), offset)) # add 0.025 s 
    subtraction =  np.concatenate((subtraction, np.zeros(len(evoked)-len(subtraction))))
    return subtraction 
    
def evoked(raw, eves, eid):
    epochs = mne.Epochs(raw, eves, eid, tmin = -0.025, proj = False, tmax = 0.701, 
                                baseline = (-0.025, 0.), reject = dict(eeg=50e-6))
    numTrial = len(epochs.events) # number of trials
    evoked = epochs.average()
    topchans = [3, 4, 7, 22, 26, 25, 30, 31]
    tiptrodes = [32, 33]
    chans = topchans  # Separately do chans = tiptrodes
    evoked_all = evoked.data[chans, :].mean(axis = 0) - evoked.data[tiptrodes, :].mean(axis=0)
    #evoked_all = evoked.data[chans, :].mean(axis = 0)
    return evoked_all, numTrial# I think, for FFR, all channels are needed

def weightedAvg(evoked, numTrial):
    numTotal = np.sum(numTrial)
    evokedAvg = np.zeros(len(evoked[1]))
    for k in range(len(numTrial)):
        evokedAvg = evokedAvg + evoked[k]*numTrial[k]/numTotal
    return evokedAvg

def highpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

########################################################################################################################################################
for subj in subjlist:
    fpath = froot + '/' + subj + '/'
    print 'Running subject', subj
    
    fs = 16384
    # extracting bdf filenames into bdfs
    bdfs = fnmatch.filter(os.listdir(fpath), subj + '_FWMK*.bdf') 

    numTrialpos = np.zeros(len(bdfs))
    numTrialneg = np.zeros(len(bdfs))
    
    if len(bdfs) >= 1:
        for k, bdf in enumerate(bdfs):                
            pos, neg, numpos, numneg, length, flags = processing(fpath, bdf, [1, 2], fs)
            
            if k == 0:
                evoked_poss = np.zeros((len(bdfs), length))
                evoked_negs = np.zeros((len(bdfs), length))
            if flags[0]:    
                evoked_poss[k] = pos
                numTrialpos[k] = numpos
            if flags[1]:
                evoked_negs[k] = neg
                numTrialneg[k] = numneg
        
        evoked_pos = weightedAvg(evoked_poss, numTrialpos) 
        evoked_neg = weightedAvg(evoked_negs, numTrialneg) 
        
        # recorded response
        PosProb = evoked_pos[0:int(0.1*fs)]
        NegProb = evoked_neg[0:int(0.1*fs)]
        PosMasked = evoked_pos[int(0.251*fs):int(0.351*fs)]
        NegMasked = evoked_neg[int(0.251*fs):int(0.351*fs)]
        adptPos = PosProb - PosMasked
        adptNeg = NegProb - NegMasked
        adptPosHp = highpass_filter(adptPos, 400, fs, 5)
        adptNegHp = highpass_filter(adptNeg, 400, fs, 5)
        
        t = np.arange(0, len(PosProb)/float(fs), 1/float(fs))
        pl.figure(1)
        pl.subplot(411)
        pl.plot(t, PosProb, 'b')
        pl.plot(t, NegProb, 'r')
        
        pl.subplot(412)
        pl.plot(t, PosMasked, 'b')
        pl.plot(t, NegMasked, 'r')
        
        pl.subplot(413)
        pl.plot(t, adptPos, 'b')
        pl.plot(t, adptNeg, 'r')
        
        pl.subplot(414)
        pl.plot(t, adptPosHp, 'b')
        pl.plot(t, adptNegHp, 'r')
        
        # stimulus polarity dependent response
        diffProb = PosProb - NegProb
        diffMasked = PosMasked - NegMasked
        diffAdpt = adptPos - adptNeg
        diffAdptHp = adptPosHp - adptNegHp
        
        pl.figure(2)
        pl.subplot(411)
        pl.plot(t, diffProb, 'b')
        pl.xlim([0.01, 0.04])
        pl.ylim([-4e-7, 4e-7])
        
        pl.subplot(412)
        pl.plot(t, diffMasked, 'b')
        pl.xlim([0.01, 0.04])
        pl.ylim([-4e-7, 4e-7])
        
        pl.subplot(413)
        pl.plot(t, diffAdpt, 'b')
        pl.xlim([0.01, 0.04])
        pl.ylim([-4e-7, 4e-7])
        
        pl.subplot(414)
        pl.plot(t, diffAdptHp, 'b')
        pl.xlim([0.01, 0.04])
        pl.ylim([-4e-7, 4e-7])
        
        # frequency component of polarity dependent adapted component
        diffAdpt_fft = np.fft.fft(diffAdptHp)
        magAdpt = np.abs(diffAdpt_fft)
        phase = np.angle(diffAdpt_fft)
        freq = np.linspace(0, fs, len(diffAdpt_fft))
        pl.figure(3)
        pl.plot(freq, magAdpt)
        
        # PLV calculation: between stimulus and evoked response
        fstim = 500
        stim = np.sin(2*np.pi*fstim*t)
        anl_stim = hilbert(stim)
        anl_resp = hilbert(diffAdptHp)
        pAgl_stim = np.angle(anl_stim)
        pAgl_resp = np.angle(anl_resp)
        plv = np.abs(np.sum(np.exp(1j*(pAgl_stim - pAgl_resp))))/len(pAgl_stim) # plv across time works for narrow-band 
        
        # PLV calculations across trials
        mtplv()
    else:
        RuntimeError("No bdf files found!")