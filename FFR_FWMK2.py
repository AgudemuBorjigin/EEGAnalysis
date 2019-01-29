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
from scipy.io import savemat  
from mne.time_frequency import tfr_multitaper
from mne.connectivity import spectral_connectivity 

def processing(fpath, bdf, eid, fs): # eid = [id_475_pos, id_475_neg, id_500_pos, id_500_neg, id_525_pos, id_525_neg,]
    rawtemp, evestemp = bs.importbdf(fpath + bdf)
    raw = rawtemp
    evestemp[:, 1] = np.mod(evestemp[:, 1], 256) 
    evestemp[:, 2] = np.mod(evestemp[:, 2], 256)
    raw.filter(l_freq = 400, h_freq=1300) # adjust this range 
    evoked_pos = []
    evoked_neg = []
    
    numpos = []
    numneg = []
    
    flags = [False, False]
    if eid[0] in evestemp[:,2]:
        evoked_pos, numpos, epochs_prob_pos, epochs_masked_pos = evoked(raw, evestemp, eid[0])
#        subtraction = offsetSub(evoked_pos, fs)
#        evoked_pos = evoked_pos - subtraction
#        evoked_pos = evoked_pos[int(0.0266*fs):int((0.0266+0.395)*fs)] # starting from time 0 to the end of masked stimulus   
        flags[0] = True;
        length = len(evoked_pos)
    if eid[1] in evestemp[:,2]:
        evoked_neg, numneg, epochs_prob_neg, epochs_masked_neg = evoked(raw, evestemp, eid[1])
#        subtraction = offsetSub(evoked_neg, fs)
#        evoked_neg = evoked_neg - subtraction 
#        evoked_neg = evoked_neg[int(0.0266*fs):int((0.0266+0.395)*fs)]   
        flags[1] = True;
        length = len(evoked_neg)
    
    return evoked_pos, evoked_neg, numpos, numneg, length, flags, epochs_prob_pos, epochs_masked_pos, epochs_prob_neg, epochs_masked_neg

def offsetSub(evoked, fs):
    offset = evoked[int(0.525*fs):int(0.555*fs)+1] # add 0.025 s
    subtraction = np.concatenate((np.zeros(int(0.275*fs)), offset)) # add 0.025 s 
    subtraction =  np.concatenate((subtraction, np.zeros(len(evoked)-len(subtraction))))
    return subtraction 
    
def evoked(raw, eves, eid):
    epochs = mne.Epochs(raw, eves, eid, tmin = 0.0016, proj = False, tmax = 0.0016 + 0.351, 
                                baseline = (None, None), reject = dict(eeg=50e-6)) # baseline correction is not that necessary since the data was already high-passed
    epochs_prob = mne.Epochs(raw, eves, eid, tmin = 0.0016, proj = False, tmax = 0.1 + 0.0016, 
                                baseline = (None, None), reject = dict(eeg=50e-6))
    epochs_masked = mne.Epochs(raw, eves, eid, tmin = 0.251 + 0.0016, proj = False, tmax = 0.0016 + 0.35105, 
                                baseline = (None, None), reject = dict(eeg=50e-6)) # picked 0.35105 instead of 0.351 to equalized the number of points of epochs_prob and epochs_masked 
    data_prob = epochs_prob.get_data()
    data_masked = epochs_masked.get_data()
    
    numTrial = len(epochs.events) # number of trials
    evoked = epochs.average()
    topchans = [3, 4, 7, 22, 26, 25, 30, 31]
    tiptrodes = [32, 33]
    chans = topchans  # Separately do chans = tiptrodes
    evoked_all = evoked.data[chans, :].mean(axis = 0) - evoked.data[tiptrodes, :].mean(axis=0)
    #evoked_all = evoked.data[chans, :].mean(axis = 0)
    return evoked_all, numTrial, data_prob, data_masked# I think, for FFR, all channels are needed

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

def diff(epochs_pos, epochs_neg):
    numEvntsPos = epochs_pos.shape
    numEvntsNeg = epochs_neg.shape
    if numEvntsPos[0] > numEvntsNeg[0]:
        subt = epochs_pos[0:numEvntsNeg[0], :, :] - epochs_neg
    else:
        subt = epochs_pos - epochs_neg[0:numEvntsPos[0], :, :]
    return subt

def diffAdption(data_prob_pos, data_prob_neg, data_masked_pos, data_masked_neg):
    data_prob_diff = diff(data_prob_pos, data_prob_neg)
    data_masked_diff = diff(data_masked_pos, data_masked_neg)
    data_adpt_diff = diff(data_prob_diff, data_masked_diff)
    numtrials_adpt = data_adpt_diff.shape
    return data_adpt_diff, numtrials_adpt[0]

def summ(epochs_pos, epochs_neg):
    numEvntsPos = epochs_pos.shape
    numEvntsNeg = epochs_neg.shape
    if numEvntsPos[0] > numEvntsNeg[0]:
        summation = epochs_pos[0:numEvntsNeg[0], :, :] + epochs_neg
    else:
        summation = epochs_pos + epochs_neg[0:numEvntsPos[0], :, :]
    return summation, min(numEvntsPos[0], numEvntsNeg[0])
 
def evokedExt(froot, subject, trigNum, fs):
    fpath = froot + '/' + subject + '_FWMK/'
    print 'Running subject', subject
    
    # extracting bdf filenames into bdfs
    bdfs = fnmatch.filter(os.listdir(fpath), subject + '_FWMK*.bdf') 

    numTrialpos = np.zeros(len(bdfs))
    numTrialneg = np.zeros(len(bdfs))
    
    if len(bdfs) >= 1:
        for k, bdf in enumerate(bdfs):                
            pos, neg, numpos, numneg, length, flags, epochs_prob_pos, epochs_masked_pos, epochs_prob_neg, epochs_masked_neg = processing(fpath, bdf, trigNum, fs) #trigNum: [1, 2], for example
            if k == 0:
                evoked_poss = np.zeros((len(bdfs), length))
                evoked_negs = np.zeros((len(bdfs), length))
                data_prob_pos = epochs_prob_pos
                data_prob_neg = epochs_prob_neg
                data_masked_pos = epochs_masked_pos
                data_masked_neg = epochs_masked_neg
            if k > 0:
                data_prob_pos = np.concatenate((epochs_prob_pos, data_prob_pos), axis = 0)
                data_prob_neg = np.concatenate((epochs_prob_neg, data_prob_neg), axis = 0)
                data_masked_pos = np.concatenate((epochs_masked_pos, data_masked_pos), axis = 0)
                data_masked_neg = np.concatenate((epochs_masked_neg, data_masked_neg), axis = 0)
            if flags[0]:    
                evoked_poss[k] = pos
                numTrialpos[k] = numpos
            if flags[1]:
                evoked_negs[k] = neg
                numTrialneg[k] = numneg
        
        evoked_pos = weightedAvg(evoked_poss, numTrialpos) 
        evoked_neg = weightedAvg(evoked_negs, numTrialneg) 
    else:
        RuntimeError("No bdf files found!")
    return evoked_pos, evoked_neg, data_prob_pos, data_prob_neg, data_masked_pos, data_masked_neg

def freqAnsis(sig):
    sig_fft = np.fft.fft(sig)
    magSig = np.abs(sig_fft)
    phase = np.angle(sig_fft)
    return magSig, phase

def subjectProcessing(froot, subj, eid, fs):
    evoked_pos, evoked_neg, data_prob_pos, data_prob_neg, data_masked_pos, data_masked_neg = evokedExt(froot, subj, eid, fs)
    data_adpt_diff, numtrials_adpt = diffAdption(data_prob_pos, data_prob_neg, data_masked_pos, data_masked_neg)
    # saving epochs and evoked
    np.savez(froot+'/'+'epochs_evoked' + '/'+subj, evoked_pos = evoked_pos, evoked_neg = evoked_neg, data_prob_pos = data_prob_pos, data_prob_neg = data_prob_neg, data_masked_pos = data_masked_pos, data_masked_neg = data_masked_neg, data_adpt_diff = data_adpt_diff) 
   
    # evoked response analysis
    diffWhole = evoked_pos - evoked_neg
    PosProb = evoked_pos[0:int(0.1*fs)]
    NegProb = evoked_neg[0:int(0.1*fs)]
    PosMasked = evoked_pos[int(0.251*fs):int(0.351*fs)]
    NegMasked = evoked_neg[int(0.251*fs):int(0.351*fs)]
    adptPos = PosProb - PosMasked
    adptNeg = NegProb - NegMasked
    diffAdpt = adptPos - adptNeg
    sumAdpt = adptPos + adptNeg
    diffProb = PosProb - NegProb
    sumProb = PosProb + NegProb
    diffMasked = PosMasked - NegMasked
    sumMasked = PosMasked + NegMasked


    # frequency analysis
    t = np.arange(0, len(diffAdpt)/float(fs), 1/float(fs))
    mag500, phase500 = freqAnsis(diffAdpt[np.logical_and(t > chop, t < chop2)])
    freq = np.linspace(0, fs, len(mag500))
    pl.figure(num = 1)
#    pl.plot(t, diffWhole, label = 'prob')
    pl.plot(t, diffAdpt, label = 'Neural')
    pl.figure(num = 2)
    pl.plot(freq, mag500)
    
    return data_adpt_diff, numtrials_adpt
########################################################################################################################################################
OS = 'Ubuntu'

if OS == 'Ubuntu':
    froot = '/media/agudemu/Storage/Data/EEG/TFS'
else:
    froot = '/Users/baoagudemu1/Desktop/Lab/EEG-Python/TFS'
subjectList = ['S031', 'S117', 'S123', 'S128', 'S132', 'S149', 'S185', 'S187', 'S191', 'S194', 'S196', 'S197'] #'S078', 'S199'
fs = 16384
chop = 0.0e-3
chop2 = 22.e-3

plvs = np.zeros(len(subjectList))
freqs = np.zeros(len(subjectList))
trialNums = np.zeros(len(subjectList))

for k, subj in enumerate(subjectList):
    data_adpt_diff, numtrials_adpt = subjectProcessing(froot, subj, [1, 2], fs)
    
    # phase locking analysis
    params = dict(Fs = 16384, tapers = [1, 1], fpass = [400, 600], itc = 0)
    # plv to TFS
    data_adpt_diff = np.transpose(data_adpt_diff, (1, 0, 2)) # switching the first and second columns
    plv, f = mtplv(data_adpt_diff, params)
    index = np.where(plv[30,] == max(plv[30,])) # 31st channel (Fz)
    plv_31 = plv[30,]
    plv_31_max = plv_31[index]
    f_max = f[index]
    dimen = data_adpt_diff.shape
    numtrials_adpt = dimen[1]
    # saving plvs
    dictMat = {"plv_31": plv_31_max, "f": f_max, "Fs": fs, "trialNum": numtrials_adpt}
    savemat(subj, dictMat)
    # collecting individual plvs into array for all subjects
    plvs[k] = plv_31_max
    freqs[k] = f_max
    trialNums[k] = numtrials_adpt
dictDataArray = {"subjects": subjectList, "plvs": plvs, "freqs": freqs, "trialNums": trialNums}
savemat('plvs_31', dictDataArray)
#plv = spectral_connectivity(data_adpt_diff, method = 'plv')
#
##phi = np.asarray([phase475[10],  phase500[11], phase525[12]]) # check the index manually from the magnitude response
##f = np.asarray([475, 500, 525])
##delays = np.diff(np.unwrap(phi - 2*np.pi*f*chop)) * 1000./ (2*np.pi*np.diff(f))
##print delays
#
#dict500 = {"adpt500": diffAdpt, "evokedPos500": evoked_pos, "evokedNeg500": evoked_neg}
#savemat('evoekd500', dict500)