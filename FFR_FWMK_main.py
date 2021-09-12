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
#from mne.time_frequency import tfr_multitaper
#from mne.connectivity import spectral_connectivity 

##############################################################################################################################################
#pre-processing
##############################################################################################################################################

########################################################################################  
def subjectProcessing(froot, subj, eid, fs):
    evoked_pos, evoked_neg, data_prob_pos, data_prob_neg, data_masked_pos, data_masked_neg, data_masker_pos, data_masker_neg = evokedExtraction(froot, subj, eid, fs)
    # data extraction for adapted component
    data_adpt_diff, numtrials_adpt, data_prob_diff, data_masker_diff = diffAdption(data_prob_pos, data_prob_neg, data_masked_pos, data_masked_neg)
    # saving epochs and evoked
    np.savez(froot+'/'+'epochs_evoked_FCz' + '/'+subj, evoked_pos = evoked_pos, evoked_neg = evoked_neg, data_prob_pos = data_prob_pos, data_prob_neg = data_prob_neg, data_masked_pos = data_masked_pos, data_masked_neg = data_masked_neg, data_masker_pos = data_masker_pos, data_masker_neg = data_masker_neg, data_adpt_diff = data_adpt_diff) 
    data_sumMasker, _ = summ(data_masker_pos, data_masker_neg)
    return evoked_pos, evoked_neg, data_sumMasker, data_adpt_diff, numtrials_adpt
########################################################################################
    
def diffAdption(data_prob_pos, data_prob_neg, data_masked_pos, data_masked_neg):
    data_prob_diff = diff(data_prob_pos, data_prob_neg)
    data_masked_diff = diff(data_masked_pos, data_masked_neg)
    data_adpt_diff = diff(data_prob_diff, data_masked_diff)
    numtrials_adpt = data_adpt_diff.shape
    return data_adpt_diff, numtrials_adpt[0], data_prob_diff, data_masked_diff

def diff(epochs_pos, epochs_neg):
    numEvntsPos = epochs_pos.shape
    numEvntsNeg = epochs_neg.shape
    if numEvntsPos[0] > numEvntsNeg[0]:
        subt = epochs_pos[0:numEvntsNeg[0], :, :] - epochs_neg
    else:
        subt = epochs_pos - epochs_neg[0:numEvntsPos[0], :, :]
    return subt

def summ(epochs_pos, epochs_neg):
    numEvntsPos = epochs_pos.shape
    numEvntsNeg = epochs_neg.shape
    if numEvntsPos[0] > numEvntsNeg[0]:
        summation = epochs_pos[0:numEvntsNeg[0], :, :] + epochs_neg
    else:
        summation = epochs_pos + epochs_neg[0:numEvntsPos[0], :, :]
    return summation, min(numEvntsPos[0], numEvntsNeg[0])

################################################################################
# taking the grand average of the evokeds from across bdf files for each subject 
# also concatenating all the epochs from across the bdf files into one variable
def evokedExtraction(froot, subject, trigNum, fs):
    fpath = froot + '/' + subject + '/'
    print 'Running subject', subject
    
    # extracting bdf filenames into bdfs
    bdfs = fnmatch.filter(os.listdir(fpath), subject + '_FWMK*.bdf') 

    numTrialpos = np.zeros(len(bdfs))
    numTrialneg = np.zeros(len(bdfs))
    
    if len(bdfs) >= 1:
        for k, bdf in enumerate(bdfs):                
            pos, neg, numpos, numneg, length, flags, epochs_prob_pos, epochs_masked_pos, epochs_masker_pos, epochs_prob_neg, epochs_masked_neg, epochs_masker_neg = processing(fpath, bdf, trigNum, fs)
            if k == 0:
                evoked_poss = np.zeros((len(bdfs), length))
                evoked_negs = np.zeros((len(bdfs), length))
                data_prob_pos = epochs_prob_pos
                data_prob_neg = epochs_prob_neg
                data_masked_pos = epochs_masked_pos
                data_masked_neg = epochs_masked_neg
                data_masker_pos = epochs_masker_pos
                data_masker_neg = epochs_masker_neg
            if k > 0:
                data_prob_pos = np.concatenate((epochs_prob_pos, data_prob_pos), axis = 0)
                data_prob_neg = np.concatenate((epochs_prob_neg, data_prob_neg), axis = 0)
                data_masked_pos = np.concatenate((epochs_masked_pos, data_masked_pos), axis = 0)
                data_masked_neg = np.concatenate((epochs_masked_neg, data_masked_neg), axis = 0)
                data_masker_pos = np.concatenate((epochs_masker_pos, data_masker_pos), axis = 0)
                data_masker_neg = np.concatenate((epochs_masker_neg, data_masker_neg), axis = 0)
            if flags[0]:    
                evoked_poss[k] = pos
                numTrialpos[k] = numpos
            if flags[1]:
                evoked_negs[k] = neg
                numTrialneg[k] = numneg
        # average weighted across bdf files based on the number of the type of events (pos or neg) within each file
        evoked_pos = weightedAvg(evoked_poss, numTrialpos) 
        evoked_neg = weightedAvg(evoked_negs, numTrialneg) 
    else:
        RuntimeError("No bdf files found!")
    return evoked_pos, evoked_neg, data_prob_pos, data_prob_neg, data_masked_pos, data_masked_neg, data_masker_pos, data_masker_neg

def weightedAvg(evoked, numTrial):
    numTotal = np.sum(numTrial)
    evokedAvg = np.zeros(len(evoked[1]))
    for k in range(len(numTrial)):
        evokedAvg = evokedAvg + evoked[k]*numTrial[k]/numTotal
    return evokedAvg

# extracts the average and the raw data of segments of prob, masker, and the masked
# the average is for FFT analysis, the raw data is for plv analysis
def processing(fpath, bdf, eid, fs): # eid = [id_475_pos, id_475_neg, id_500_pos, id_500_neg, id_525_pos, id_525_neg,]
    raw, eves = bs.importbdf(fpath + bdf)
    raw.info['bads'] += [ 'A28', 'A3', 'A6', 'A24', 'A7', 'A25', 'A27']
    eves[:, 1] = np.mod(eves[:, 1], 256) 
    eves[:, 2] = np.mod(eves[:, 2], 256)
    raw.filter(l_freq = 400, h_freq=1300) # adjust this range 
    evoked_pos = []
    evoked_neg = []   
    
    numpos = []
    numneg = []
    
    flags = [False, False]
    if eid[0] in eves[:,2]:
        evoked_pos, numpos, epochs_prob_pos, epochs_masked_pos, epochs_masker_pos = evoked(raw, eves, eid[0])   
        flags[0] = True;
        length = len(evoked_pos)
    if eid[1] in eves[:,2]:
        evoked_neg, numneg, epochs_prob_neg, epochs_masked_neg, epochs_masker_neg = evoked(raw, eves, eid[1])
        flags[1] = True;
        length = len(evoked_neg)
    
    return evoked_pos, evoked_neg, numpos, numneg, length, flags, epochs_prob_pos, epochs_masked_pos, epochs_masker_pos, epochs_prob_neg, epochs_masked_neg, epochs_masker_neg    

def evoked(raw, eves, eid):
    eegReject = 50e-6
    epochs = mne.Epochs(raw, eves, eid, tmin = 0.0016, proj = False, tmax = 0.0016 + 0.351, 
                                baseline = (None, None), reject = dict(eeg=eegReject)) 
    # baseline correction is not that necessary since the data was already high-passed
    epochs_prob = mne.Epochs(raw, eves, eid, tmin = 0.0016, proj = False, tmax = 0.1 + 0.0016, 
                                baseline = (None, None), reject = dict(eeg=eegReject))
    epochs_masked = mne.Epochs(raw, eves, eid, tmin = 0.251 + 0.0016, proj = False, tmax = 0.0016 + 0.35105, 
                                baseline = (None, None), reject = dict(eeg=eegReject)) 
    # picked 0.35105 instead of 0.351 to equalized the number of points of epochs_prob and epochs_masked 
    epochs_masker = mne.Epochs(raw, eves, eid, tmin = 0.15 + 0.0016, proj = False, tmax = 0.0016 + 0.25, 
                                baseline = (None, None), reject = dict(eeg=eegReject))
    data_prob = epochs_prob.get_data()
    data_masked = epochs_masked.get_data()
    data_masker = epochs_masker.get_data()
    
    numTrial = len(epochs.events) # number of trials
    evoked = epochs.average()
    evoked_all = evoked.data[topchans, :].mean(axis = 0)
    #tiptrodes = [32, 33]
    #evoked_all = evoked.data[chans, :].mean(axis = 0) - evoked.data[tiptrodes, :].mean(axis=0)
    return evoked_all, numTrial, data_prob, data_masked, data_masker

########################################################################################################################################################
# Data analysis
########################################################################################################################################################
# evoked response analysis
def diffSumEvd(evoked_pos, evoked_neg):
    diffWhole = evoked_pos - evoked_neg
    
    PosProb = evoked_pos[0:int(0.1*fs)]
    NegProb = evoked_neg[0:int(0.1*fs)]
    diffProb = PosProb - NegProb
    sumProb = PosProb + NegProb
    
    PosMasked = evoked_pos[int(0.251*fs):int(0.351*fs)]
    NegMasked = evoked_neg[int(0.251*fs):int(0.351*fs)]
    diffMasked = PosMasked - NegMasked
    sumMasked = PosMasked + NegMasked
    
    PosMasker = evoked_pos[int(0.15*fs):int(0.25*fs)]
    NegMasker = evoked_neg[int(0.15*fs):int(0.25*fs)]
    diffMasker = PosMasker - NegMasker
    sumMasker = PosMasker + NegMasker
   
    adptPos = PosProb - PosMasked
    adptNeg = NegProb - NegMasked
    diffAdpt = adptPos - adptNeg
    sumAdpt = adptPos + adptNeg
    return diffWhole, diffProb, sumProb, diffMasked, sumMasked, diffMasker, sumMasker, diffAdpt, sumAdpt

def fftPlot(avg, fs, chop, chop2, stimType):
    # frequency analysis
    t = np.arange(0, len(avg)/float(fs), 1/float(fs))
    mag, phase500 = freqAnsis(avg[np.logical_and(t > chop, t < chop2)])
    y = avg[np.logical_and(t > chop, t < chop2)]
    x = t[np.logical_and(t > chop, t < chop2)]
    freq = np.linspace(0, fs/4, len(mag)/4)
    
    # plot figure
    fig = pl.figure(figsize = (20, 5))
    ax = fig.add_subplot(211)
    titleStr = subj + '_' + stimType
    pl.title(titleStr, fontsize=14)
    pl.xlabel('Time (s)', fontsize=14)
    pl.ylabel('Evoked response', fontsize=14)
    ax.plot(x, y)
    
    ax = fig.add_subplot(212)
    peak = max(mag[0:len(mag)/4])
    index = np.where(mag[0:len(mag)/4] == peak)
    ax.plot(freq, mag[0:len(mag)/4])
    ax.plot(freq[index], peak, 'r+', markersize = 12, linewidth = 8)
    ax.annotate(repr(peak), xy = (freq[index], peak), xytext = (freq[index], peak))
    pl.xlabel('Freq (Hz)', fontsize=14)
    pl.ylabel('Amplitude', fontsize=14)
    pl.savefig(froot + '/FWMKfigures_FCz/' + titleStr + '.png')
    return peak, index

def freqAnsis(sig):
    sig_fft = np.fft.fft(sig)
    magSig = np.abs(sig_fft)
    phase = np.angle(sig_fft)
    return magSig, phase

def plvAnalysis(epochs, Type):
    # phase locking analysis
    params = dict(Fs = 16384, tapers = [1, 1], fpass = [400, 600], itc = 0)
    # plv to TFS
    epochs = np.transpose(epochs, (1, 0, 2)) # switching the first and second columns
    plv, f = mtplv(epochs, params)
    index = np.where(plv[topchans[0],] == max(plv[topchans[0],])) 
    plv_max = max(plv[topchans[0],])
    f_max = f[index]
    dimen = epochs.shape
    numtrials = dimen[1]
    return plv_max, f_max, numtrials
    
########################################################################################################################################################
OS = 'Ubuntu'

if OS == 'Ubuntu':
    froot = '/media/agudemu/Storage/Data/EEG/FFR'
else:
    froot = '/Users/baoagudemu1/Desktop/Lab/EEG-Python/FFR'
    
subjectList = ['S194']
topchans = [30] #CHANGE AS NEEDED

fs = 16384
chop = 0.0e-3
chop2 = 50e-3

for k, subj in enumerate(subjectList):
    evoked_pos, evoked_neg, data_sumMasker, data_adpt_diff, numtrials_adpt = subjectProcessing(froot, subj, [1, 2], fs)
    diffWhole, diffProb, sumProb, diffMasked, sumMasked, diffMasker, sumMasker, diffAdpt, sumAdpt = diffSumEvd(evoked_pos, evoked_neg)
    peak_masker_sum, _ = fftPlot(sumMasker, fs, chop, chop2, 'MaskerSum')
    peak_adpt_diff, _ = fftPlot(diffAdpt, fs, chop, chop2, 'AdptDiff')
    # RMS
    PosMaskerOnset = evoked_pos[int(0.152*fs):int(0.159*fs)]
    NegMaskerOnset = evoked_neg[int(0.152*fs):int(0.159*fs)]
    maskerOnset = PosMaskerOnset + NegMaskerOnset
    rmsOnset = np.sqrt(np.mean(np.multiply(maskerOnset, maskerOnset)))
#   peak_masker_diff, _ = fftPlot(diffMasker, fs, chop, chop2, 'MaskerDiff')
#   peak_prob_diff, _ = fftPlot(diffProb, fs, chop, chop2, 'ProbDiff')
#   PosMasker = evoked_pos[int(0.15*fs):int(0.25*fs)]
#   NegMasker = evoked_neg[int(0.15*fs):int(0.25*fs)]
#   peak_abr_pos, _ = fftPlot(PosMasker, fs, 0, 12e-3, 'ABR to positive masker')
#   peak_abr_neg, _ = fftPlot(NegMasker, fs, 0, 12e-3, 'ABR to negative masker')
    
#   plv_masker_sum, _, _ = plvAnalysis(data_sumMasker, 'MaskerSum')
#   plv_adpt_diff, _, _ = plvAnalysis(data_adpt_diff, 'AdptDiff')
#   plv = spectral_connectivity(data_adpt_diff, method = 'plv')
#   phi = np.asarray([phase475[10],  phase500[11], phase525[12]]) # check the index manually from the magnitude response
#   f = np.asarray([475, 500, 525])
#   delays = np.diff(np.unwrap(phi - 2*np.pi*f*chop)) * 1000./ (2*np.pi*np.diff(f))
#   print delays