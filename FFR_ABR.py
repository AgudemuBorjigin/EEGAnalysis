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
    raw.filter(l_freq = 100, h_freq=2000) # ABR is broadband
    evoked_pos = []
    evoked_neg = []
    
    numpos = []
    numneg = []
    
    flags = [False, False]
    if eid[0] in evestemp[:,2]:
        evoked_pos, numpos, data_masker_pos = evoked(raw, evestemp, eid[0])   
        flags[0] = True;
        length = len(evoked_pos)
    if eid[1] in evestemp[:,2]:
        evoked_neg, numneg, data_masker_neg = evoked(raw, evestemp, eid[1]) 
        flags[1] = True;
        length = len(evoked_neg)
    return evoked_pos, evoked_neg, numpos, numneg, length, flags, data_masker_pos, data_masker_neg

def offsetSub(evoked, fs):
    offset = evoked[int(0.525*fs):int(0.555*fs)+1] # add 0.025 s
    subtraction = np.concatenate((np.zeros(int(0.275*fs)), offset)) # add 0.025 s 
    subtraction =  np.concatenate((subtraction, np.zeros(len(evoked)-len(subtraction))))
    return subtraction 
    
def evoked(raw, eves, eid):
    epochs_masker = mne.Epochs(raw, eves, eid, tmin = 0.15 + 0.0016, proj = False, tmax = 0.0016 + 0.162, 
                                baseline = (None, None), reject = dict(eeg=50e-6))
    data_masker = epochs_masker.get_data()
    
    numTrial = len(epochs_masker.events) # number of trials
    evoked = epochs_masker.average()
    topchans = [3, 4, 7, 22, 26, 25, 30, 31]
    tiptrodes = [32, 33]
    chans = topchans  # Separately do chans = tiptrodes
    evoked_all = evoked.data[chans, :].mean(axis = 0) - evoked.data[tiptrodes, :].mean(axis=0)
    #evoked_all = evoked.data[chans, :].mean(axis = 0)
    return evoked_all, numTrial, data_masker# I think, for FFR, all channels are needed

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
            pos, neg, numpos, numneg, length, flags, epochs_masker_pos, epochs_masker_neg = processing(fpath, bdf, trigNum, fs) #trigNum: [1, 2], for example
            if k == 0:
                evoked_poss = np.zeros((len(bdfs), length))
                evoked_negs = np.zeros((len(bdfs), length))
                data_masker_pos = epochs_masker_pos
                data_masker_neg = epochs_masker_neg
            if k > 0:
                data_masker_pos = np.concatenate((epochs_masker_pos, data_masker_pos), axis = 0)
                data_masker_neg = np.concatenate((epochs_masker_neg, data_masker_neg), axis = 0)
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
    return evoked_pos, evoked_neg, data_masker_pos, data_masker_neg

def freqAnsis(sig):
    sig_fft = np.fft.fft(sig)
    magSig = np.abs(sig_fft)
    phase = np.angle(sig_fft)
    return magSig, phase

def subjectProcessing(froot, subj, eid, fs):
    evoked_pos, evoked_neg, data_masker_pos, data_masker_neg = evokedExt(froot, subj, eid, fs)
    data_masker_sum, numtrials_masker = summ(data_masker_pos, data_masker_neg)
    # saving epochs and evoked
    np.savez(froot+'/'+'epochs_evoked_ABR' + '/'+subj, evoked_pos = evoked_pos, evoked_neg = evoked_neg, data_masker_pos = data_masker_pos, data_masker_neg = data_masker_neg, data_masker_sum = data_masker_sum) 
   
    # evoked response analysis
    diffMasker = evoked_pos - evoked_neg
    sumMasker = evoked_pos + evoked_neg

    # frequency analysis
    t = np.arange(0, len(sumMasker)/float(fs), 1/float(fs))
    mag500, phase500 = freqAnsis(sumMasker[np.logical_and(t > chop, t < chop2)])
    freq = np.linspace(0, fs, len(mag500))
    pl.figure(num = 1)
#    pl.plot(t, evoked_pos, label = 'Pos')
#    pl.plot(t, evoked_neg, label = 'Neg')
    pl.plot(t, sumMasker, label = 'Sum')
    pl.legend()
    pl.figure(num = 2)
    pl.plot(freq, mag500)
    
    return data_masker_sum, numtrials_masker
########################################################################################################################################################
OS = 'Ubuntu'

if OS == 'Ubuntu':
    froot = '/media/agudemu/Storage/Data/EEG/FFR'
else:
    froot = '/Users/baoagudemu1/Desktop/Lab/EEG-Python/TFS'
subjectList = ['S025', 'S031', 'S043', 'S051', 'S072', 'S075', 'S078', 'S084', 'S117', 'S123', 'S127', 'S128', 'S132', 'S133', 'S149', 'S183', 'S185', 
               'S187', 'S191', 'S194', 'S195', 'S196', 'S197', 'S199', 'S216', 'S218'] # exlcuded S078, S123 (empty arrays were returned), S199 (memory problem)

fs = 16384
chop = 0.0e-3
chop2 = 22.e-3

plv_31s = np.zeros(len(subjectList))
plv_auds = np.zeros(len(subjectList))
trialNums_ref = np.zeros(len(subjectList))

for k, subj in enumerate(subjectList):
    data_masker_sum, numtrials_masker = subjectProcessing(froot, subj, [1, 2], fs)
    
    # phase locking analysis
    params = dict(Fs = 16384, tapers = [1, 1], fpass = [100, 2000], itc = 0)
    # plv to ABR (onset response)
    data_masker_sum = np.transpose(data_masker_sum, (1, 0, 2)) # switching the first and second columns
    plv, f = mtplv(data_masker_sum, params)
    plv_31 = plv[30,]
    plv_aud = plv[[4, 26, 25, 30, 31], :].mean(axis = 0)
    plv_31_avg = plv_31.mean()
    plv_aud_avg = plv_aud.mean()
    # saving plvs
    dictMat = {"plv_31": plv_31_avg, "plv_aud": plv_aud_avg, "Fs": fs, "trialNum": numtrials_masker}
    savemat(subj + '_ABR', dictMat)
    # collecting individual plvs into array for all subjects
    #plv_31s[k] = plv_31_avg
    #plv_auds[k] = plv_aud_avg
    #trialNums_ref[k] = numtrials_masker
#dictDataArray = {"subjects": subjectList, "plv_31s": plv_31s, "plv_auds": plv_auds, "trialNums_ref": trialNums_ref}
#savemat('plvs_ABR', dictDataArray)

#plv = spectral_connectivity(data_adpt_diff, method = 'plv')
#
##phi = np.asarray([phase475[10],  phase500[11], phase525[12]]) # check the index manually from the magnitude response
##f = np.asarray([475, 500, 525])
##delays = np.diff(np.unwrap(phi - 2*np.pi*f*chop)) * 1000./ (2*np.pi*np.diff(f))
##print delays
#
#dict500 = {"adpt500": diffAdpt, "evokedPos500": evoked_pos, "evokedNeg500": evoked_neg}
#savemat('evoekd500', dict500)
