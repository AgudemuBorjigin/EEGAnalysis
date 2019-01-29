"""
Created on Mon Dec  3 10:24:01 2018
@author: agudemu
"""

from anlffr.helper import biosemi2mne as bs
import mne
import numpy as np # support for large, multi-dimensional arrays and metrices
import pylab as pl
import os # it assigns its path attribute to an os-specific path module
import fnmatch # unix filename pattern matching
OS = 'Ubuntu'

if OS == 'Ubuntu':
        froot = '/media/agudemu/Storage/Data/EEG/FM'
else:
    froot = '/Users/baoagudemu1/Desktop/Lab/EEG-Python/FM'

subjlist = ['S116_TEST2']     

def processing(fpath, bdf, eid): # eid = [id_475_pos, id_475_neg, id_500_pos, id_500_neg, id_525_pos, id_525_neg,]
    rawtemp, evestemp = bs.importbdf(fpath + bdf)
    raw = rawtemp
    eves2 = evestemp.copy()
    eves2[:, 1] = np.mod(eves2[:, 1], 256) 
    eves2[:, 2] = np.mod(eves2[:, 2], 256)
    raw.filter(l_freq = 400, h_freq=1300) # adjust this range 
    evoked_475_pos = []
    evoked_475_neg = []
    evoked_500_pos = []
    evoked_500_neg = []
    evoked_525_pos = []
    evoked_525_neg = []
    num475pos = []
    num475neg = []
    num500pos = []
    num500neg = []
    num525pos = []
    num525neg = []
    flags = [False, False, False, False, False, False]
    if eid[0] in eves2[:,2]:
        evoked_475_pos, num475pos = epochsExtraction(raw, eves2, eid[0])
        flags[0] = True;
        length = len(evoked_475_pos)
    if eid[1] in eves2[:,2]:
        evoked_475_neg, num475neg = epochsExtraction(raw, eves2, eid[1])
        flags[1] = True;
        length = len(evoked_475_neg)
    if eid[2] in eves2[:,2]:
        evoked_500_pos, num500pos = epochsExtraction(raw, eves2, eid[2])
        flags[2] = True;
        length = len(evoked_500_pos)
    if eid[3] in eves2[:,2]:
        evoked_500_neg, num500neg = epochsExtraction(raw, eves2, eid[3])
        flags[3] = True;
        length = len(evoked_500_neg)
    if eid[4] in eves2[:,2]:
        evoked_525_pos, num525pos = epochsExtraction(raw, eves2, eid[4])
        flags[4] = True;
        length = len(evoked_525_pos)
    if eid[5] in eves2[:,2]:
        evoked_525_neg, num525neg = epochsExtraction(raw, eves2, eid[5])
        flags[5] = True;
        length = len(evoked_525_neg)
    return evoked_475_pos, evoked_475_neg, evoked_500_pos, evoked_500_neg, evoked_525_pos, evoked_525_neg, num475pos, num475neg, num500pos, num500neg, num525pos, num525neg, length, flags

def epochsExtraction(raw, eves, eid):
    epochs = mne.Epochs(raw, eves, eid, tmin = -0.025, proj = False, tmax = 0.125, 
                                baseline = (-0.025, 0.), reject = dict(eeg=50e-6))
    numTrial = len(epochs.events) # number of trials
    evoked = epochs.average()
    topchans = [3, 4, 7, 22, 26, 25, 30, 31]
    tiptrodes = [32, 33]
    chans = topchans  # Separately do chans = tiptrodes
    evoked_all = evoked.data[chans, :].mean(axis = 0) - evoked.data[tiptrodes, :].mean(axis=0)
    #evoked_all = evoked.data[chans, :].mean(axis = 0)
    return evoked_all, numTrial # I think, for FFR, all channels are needed

def weightedAvg(evoked, numTrial):
    numTotal = np.sum(numTrial)
    evokedAvg = np.zeros(len(evoked[1]))
    for k in range(len(numTrial)):
        evokedAvg = evokedAvg + evoked[k]*numTrial[k]/numTotal
    return evokedAvg
#    nozeros = np.nonzero(numTrial)
#    return np.sum(evoked, axis = 0)/len(nozeros[0])    

subjN = 0;
for subj in subjlist:
    subjN = subjN + 1;
    fpath = froot + '/' + subj + '/'
    print 'Running subject', subj
    
    # extracting bdf filenames into bdfs
    bdfs = fnmatch.filter(os.listdir(fpath), subj + '_FFR*.bdf') 

    numTrial475pos = np.zeros(len(bdfs))
    numTrial475neg = np.zeros(len(bdfs))
    numTrial500pos = np.zeros(len(bdfs))
    numTrial500neg = np.zeros(len(bdfs))
    numTrial525pos = np.zeros(len(bdfs))
    numTrial525neg = np.zeros(len(bdfs))

    
    if len(bdfs) >= 1:
        for k, bdf in enumerate(bdfs):                
            pos_475, neg_475, pos_500, neg_500, pos_525, neg_525, num475pos, num475neg, num500pos, num500neg, num525pos, num525neg, length, flags = processing(fpath, bdf, [1, 2, 3, 4, 5, 6])
            
            if k == 0:
                evoked_pos_475s = np.zeros((len(bdfs), length))
                evoked_neg_475s = np.zeros((len(bdfs), length))
                evoked_pos_500s = np.zeros((len(bdfs), length))
                evoked_neg_500s = np.zeros((len(bdfs), length))
                evoked_pos_525s = np.zeros((len(bdfs), length))
                evoked_neg_525s = np.zeros((len(bdfs), length))
            if flags[0]:    
                evoked_pos_475s[k] = pos_475
                numTrial475pos[k] = num475pos
            if flags[1]:
                evoked_neg_475s[k] = neg_475
                numTrial475neg[k] = num475neg
            if flags[2]:
                evoked_pos_500s[k] = pos_500
                numTrial500pos[k] = num500pos
            if flags[3]:
                evoked_neg_500s[k] = neg_500
                numTrial500neg[k] = num500neg
            if flags[4]:
                evoked_pos_525s[k] = pos_525
                numTrial525pos[k] = num525pos
            if flags[5]:
                evoked_neg_525s[k] = neg_525
                numTrial525neg[k] = num525neg
        
        evoked_pos_475 = weightedAvg(evoked_pos_475s, numTrial475pos) 
        evoked_neg_475 = weightedAvg(evoked_neg_475s, numTrial475neg) 
        evoked_pos_500 = weightedAvg(evoked_pos_500s, numTrial500pos) 
        evoked_neg_500 = weightedAvg(evoked_neg_500s, numTrial500neg) 
        evoked_pos_525 = weightedAvg(evoked_pos_525s, numTrial525pos) 
        evoked_neg_525 = weightedAvg(evoked_neg_525s, numTrial525neg)
        
        evoked_475 = evoked_pos_475 - evoked_neg_475
        evoked_500 = evoked_pos_500 - evoked_neg_500
        evoked_525 = evoked_pos_525 - evoked_neg_525
        fs = 16384
        # Extra 20 ms chopped off at the beginning, to get the steady state response
        chop = 20.0e-3
        evoked_475 = evoked_475[int((0.0266 + chop)*fs):int(0.1266*fs)]
        evoked_500 = evoked_500[int((0.0266 + chop)*fs):int(0.1266*fs)]
        evoked_525 = evoked_525[int((0.0266 + chop)*fs):int(0.1266*fs)]
        
        evoked_475_fft = np.fft.fft(evoked_475)
        evoked_475_mag = np.abs(evoked_475_fft)
        evoked_475_phase = np.angle(evoked_475_fft)
        evoked_500_fft = np.fft.fft(evoked_500)
        evoked_500_mag = np.abs(evoked_500_fft)
        evoked_500_phase = np.angle(evoked_500_fft)
        evoked_525_fft = np.fft.fft(evoked_525)
        evoked_525_mag = np.abs(evoked_525_fft)
        evoked_525_phase = np.angle(evoked_525_fft)
        #evoked_500_phase[40], 
        phi = np.asarray([evoked_475_phase[38],  evoked_525_phase[42]]) # check the index manually from the magnitude response
        f = np.asarray([475, 525])
        delays = np.diff(np.unwrap(phi - 2*np.pi*f*chop)) * 1000./ (2*np.pi*np.diff(f))
        print delays
        
        
        freq = np.linspace(0, fs, len(evoked_475_fft))
        pl.figure()
        pl.plot(freq, evoked_475_mag)
        pl.show()
        pl.figure()
        pl.plot(freq, evoked_500_mag)
        pl.show()
        pl.figure()
        pl.plot(freq, evoked_525_mag)
        pl.show()
        
    else:
        RuntimeError("No bdf files found!")