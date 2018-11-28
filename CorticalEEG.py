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
import os # it assigns its path attribute to an os-specific path module
import fnmatch # unix filename pattern matching
from scipy.signal import butter, lfilter

stimulus = 'ITD'
OS = 'Ubuntu'

subjlist = ['S043']     
#subjlist = ['S025', 'S028', 'S031', 'S043', 'S046', 'S072', 'S075', 'S078', 'S083', 'S117', 'S119', 'S123', 'S127', 'S128', 'S132', 
# S133, S139, S140, S143, S144, S145, S149]
# S084 and S135 need different projecions

if stimulus == 'ITD':
    if OS == 'Ubuntu':
        froot = '/media/agudemu/Storage/Data/EEG/ITD'
    else:
        froot = '/Users/baoagudemu1/Desktop/Lab/Experiment/DataAnalysis/Data'    
elif stimulus == 'Atten':
    if OS == 'Ubuntu':
        froot = '/media/agudemu/Storage/Data/EEG/Atten'
    else:
        froot = '/Users/baoagudemu1/Desktop/Lab/EEG-Python/Atten'
elif stimulus == 'FM':
    if OS == 'Ubuntu':
        froot = '/media/agudemu/Storage/Data/EEG/FM'
    else:
        froot = '/Users/baoagudemu1/Desktop/Lab/EEG-Python/FM'


for subj in subjlist:
    fpath = froot + '/' + subj + '/'
    print 'Running subject', subj
    
    # load data and read event channels
    rawlist = []
    evelist = []
    # extracting bdf filenames into bdfs
    if stimulus == 'ITD':
        bdfs = fnmatch.filter(os.listdir(fpath), subj + '_ITD*.bdf') 
    elif stimulus == 'Atten':
        bdfs = fnmatch.filter(os.listdir(fpath), subj + '_Atten*.bdf') 
    elif stimulus == 'FM':
        bdfs = fnmatch.filter(os.listdir(fpath), subj + '_FM*.bdf') 
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
    # the eves are in 16 bits originally, so this operation only looks at the lower 8 bits. higher 8 bits are always 
    # high, representing button box condition. If interested in what's happening at button press, do np.floor(eves2[:,1]/256)
    eves2[:, 1] = np.mod(eves2[:, 1], 256) 
    eves2[:, 2] = np.mod(eves2[:, 2], 256)
##############################################################################################################################################    
    if stimulus == 'ITD' or stimulus == 'Atten':
        raw.filter(l_freq = 0.5, h_freq=50) # if channels are noisy, adjust the filter parameters
    elif stimulus == 'FM':
        raw.filter(l_freq = 10, h_freq=3000) # adjust this range 
    # raw.plot(events=eves2)
    
    # SSP for blinks
    blinks = find_blinks(raw, ch_name = ['A1',], l_trans_bandwidth=0.5,
                         l_freq=1.0) # A1 is the closest electrode to eyebrow
    # blink and eves2 triggers can be combined using np.concatenate((eves2, blinks), axis = 0)
    # raw.plot(events=blinks) shows the lines at eye blinks
    
    # the trigger for blinks can be chosen to be starting from 1000, just to make sure it doesn't collide with the triggers for conditions
    epochs_blinks = mne.Epochs(raw, blinks, 998, tmin = -0.05, tmax = 0.15, 
                               proj = False, baseline = (-0.05, 0), 
                               reject=dict(eeg=500e-6)) 
    evoked_blinks = epochs_blinks.average()
    evoked_blinks_data = evoked_blinks.data[np.arange(32),:]
    # PCA is only applied to the epochs around eye blinks. Since the eye blinks are 
    # contributing the most to the variance within this chunk of window, 
    # the first PCA (first eigenvector) is going to be due to the eye blink 
    # for sure and removed. If the PCA was performed on the whole samples, we wouldn't
    # know which vector is going to be linked to the eye blink
    # for "n_eeg", it's recommended to remove only the biggest projection, which is eye blinks in this case
    # greater n_eeg removes more nueral data, which is not favorable
    n_eeg = 4
    blink_projs = compute_proj_epochs(epochs_blinks, n_grad=0,
                                      n_mag=0, n_eeg=n_eeg,
                                      verbose='DEBUG')  
        
    #raw.add_proj(blink_projs) # adding all projections
    raw.add_proj([blink_projs[0], blink_projs[2]]) # raw.del_proj()   
    # raw.plot_projs_topomap() shows the 4 max PCAs (eigenvectors)
    # raw.plot(events = blinks, show_options = True) could show the options for applying different projections
    
    # if channels are too noisy, play with n_eeg, if the second variance is acting more than
    # the first, that means the channels are contaminated not just by the eye blinks, but also
    # from other sources

    # MANUALLY SELECT PROJECTIONS BY PLOTTING raw.plot_projs_topomap
    # REMOVE EXTRA PROJS USING raw.del_proj -- Remember index starts at 0
###########################################################################################################################################
    def noiseFloorEstimate(t, evoked):
        index = np.where(t>0)
        index1 = index[0]
        index1 = index1[0]
        index = np.where(t<0.3) # CHANGE AS NEEDED FOR DIFFERENT SUBJECTS, 0.3
        index2 = index[0]
        index2 = index2[-1]
        
        noiseFloor = evoked[0:index1].mean(axis=0)
        return noiseFloor, index1, index2
############################################################################################################################################
    def itc_normalization(itc, freqs, t):
        if stimulus == 'ITD':
            freqSub = np.where(freqs<20) # CHANGE AS NEEDED FOR DIFFERENT SUBJECTS, refer to time-freq plot of itc, 23
        elif stimulus == 'Atten':
            freqSub = np.where(freqs<50) # CHANGE AS NEEDED FOR DIFFERENT SUBJECTS, refer to time-freq plot of itc, 23
            
        itc_ave = itc[freqSub[0], :].mean(axis=0)
        noiseFloor, index1, index2 = noiseFloorEstimate(t, itc_ave)
        itc_ave = itc_ave - noiseFloor
        
        firstPeakAmp = np.absolute(np.max(itc_ave[index1:index2])) 
        itc_norm = itc_ave/firstPeakAmp
        return itc_norm
############################################################################################################################################
    def itc(t, epochs, cond):
        # computation of inter-trial-coherence (itc)
        freqs = np.arange(1., 50., 1.) # CHANGE AS NEEDED
        n_cycles = freqs/4. # time resolution is 0.25 s (1 s has "freq" number of cycles)
        time_bandwidth = 2.0 # number of taper = time_bandwidth product - 1 
        # usually n_cycles and time_bandwidth are fixed, which determines the frequency resolution 
        power, itc = tfr_multitaper(epochs, freqs = freqs,  n_cycles = n_cycles,
                   time_bandwidth = time_bandwidth, return_itc = True, n_jobs = 4)
        # itc.plot([channel number], mode = 'mean')
        itc_copy = itc.copy()
        # itc_copy.plot_topo(baseline = (-0.5, 0), mode = 'zscore') 
        itc_data = itc_copy.data
         # averaging across channels
        itc_data_mean_all = itc_data.mean(axis = 0) # CHANGE AS NEEDED: average across all channels 
        itc_data_mean_auditory = itc_data[[4, 26, 25, 30, 31], :, :].mean(axis = 0) # these channels are auditory 
        
        np.savez(froot+'/'+'itcs' + '/'+subj+'_'+'itc'+str(cond), itc = itc_data, t = t, freqs = freqs, itc_avg_all = itc_data_mean_all, itc_avg_auditory = itc_data_mean_auditory); # itc_data is the most time consuming variable
        # npzFile = np.load(fpath+'/'+'itc_power.npz'), npzFiles.files, npzFile['itc']
        # if bad channels were added manually, select good channels
        power_copy = power.copy()
        power_data = power_copy.data
        np.savez(froot+'/'+'powers' + '/'+subj+'_'+'power'+str(cond), power = power_data, t = t, freqs = freqs);
        
        return itc_data_mean_all, itc_data_mean_auditory
    
    def plot_spectrogram(itc_data_mean, t, freqs, cond):
        fig, ax = pl.subplots()
        ax.imshow(itc_data_mean, interpolation='bicubic', aspect='auto', 
                  origin='lower', cmap='RdBu_r', vmin = 0, vmax = 0.38) 
        # ax.set_xticks(np.arange(t[0], t[-1]+0.5,0.5));
        ax.set_xticks(np.arange(0, len(t), len(t)/6));
        ax.set_xticklabels(['-0.5', '0', '0.5', '1', '1.5', '2.0', '2.5'])
        ax.set_yticks(np.arange(0, len(freqs), len(freqs)/11));
        ax.set_yticklabels(['1', '5', '9', '13', '17', '21', '25', '29', '33', '37', '41', '45', '49'])
        ax.set_title('Inter-trial coherence averaged across channels:'+str(cond))
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [Hz]')
        pl.show()
        
    def peak(itc_norm, t):
        index = np.where(t>0.9)
        index1 = index[0]
        index1 = index1[0]
        index = np.where(t<1.3) # CHANGE AS NEEDED FOR DIFFERENT SUBJECTS, 0.3
        index2 = index[0]
        index2 = index2[-1]
        return max(itc_norm[index1:index2])
        
    def plot_avg(itc_norm, t, cond):
        pl.figure()
        pl.plot(t, itc_norm)
        pl.xlabel('Time (s)', fontsize=14)
        pl.ylabel('Normalized response', fontsize=14)
        pl.title('Phase locking:'+str(cond), fontsize=14)
        ax = pl.gca()
        ax.tick_params(labelsize=14)
        pl.show()
        
        peakAmp = peak(itc_norm, t)
        return peakAmp
##################################################################################################################################################    
    def normalizedN1p2(evoked, t, timeWindow):
        index = np.where(t>timeWindow[0])
        index1 = index[0]
        index1 = index1[0]
        
        index = np.where(t>timeWindow[1])
        index2 = index[0]
        index2 = index2[0]
        
        index = np.where(t>timeWindow[2])
        index3 = index[0]
        index3 = index3[0]
        
        index = np.where(t>timeWindow[3])
        index4 = index[0]
        index4 = index4[0]
        
        n1p2onset = np.abs(np.min(evoked[index1:index2]) - np.max(evoked[index1:index2]))
        n1p2ITD = np.abs(np.min(evoked[index3:index4]) - np.max(evoked[index3:index4]))
        evokedAmpNor = n1p2ITD/n1p2onset  
        return evokedAmpNor
##################################################################################################################################################            
    def auditoryAvg(evoked):
        evokedAvg = np.zeros(shape = (1, len(evoked.data[0])))
        audChanns = [4, 26, 25, 30, 31]
        for i in audChanns:
            evokedAvg = evokedAvg + evoked.data[i]
        evokedAvg = evokedAvg / len(audChanns)
        return evokedAvg[0]
##################################################################################################################################################    
    def evoked(triggerID, timeWindow):
        epochs = mne.Epochs(raw, eves2, triggerID, tmin = -0.5, proj = True, tmax = 2.5, 
                        baseline = (-0.5, 0), reject = dict(eeg=150e-6)) # change the channels as needed
        t = epochs.times
        # always start with looking at evoked (averaged) response, 
        # and see which channels are bad by using evoked.plot(picks=[30, 31]) and evoked.plot_topomap(times=[1.2]), evoked.pick_chanels()
        # add those channels to the list manually 
        # by raw.info['bads'] += ['A7', 'A6', 'A24'] if necessary
        evoked_raw = epochs.average()
        # noiseFloor, _, _, = noiseFloorEstimate(t, evoked.data[31]) # noise floor is very small, since DC has been filtered out
        # evoked_chann32 = evoked.data[31] - noiseFloor
        evokedAud = auditoryAvg(evoked_raw)
        evokedAll = evoked_raw.data.mean(axis = 0)
        #evokedAud = butter_lowpass_filter(evokedAud, 40.8, 44100, 5)
#        evoked.plot(picks = [31])
#        pl.plot(t*1e3, evokedAud*1e6)
#        pl.axvline(x = timeWindow[0]*1e3, color = 'r')
#        pl.axvline(x = timeWindow[1]*1e3, color = 'r')
#        pl.axvline(x = timeWindow[2]*1e3, color = 'r')
#        pl.axvline(x = timeWindow[3]*1e3, color = 'r')
        ampAud = normalizedN1p2(evokedAud, t, timeWindow)
        ampAll = normalizedN1p2(evokedAll, t, timeWindow)
        return epochs, ampAll, ampAud
##################################################################################################################################################        
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_lowpass_filter(data, cutoff, fs, order):
        b, a = butter_lowpass(cutoff, fs, order)
        y = lfilter(b, a, data)
        return y
##################################################################################################################################################    
    def fft_evoked(evoked, t, fs, timeWindow):
        index = np.where(t>timeWindow[0])
        index1 = index[0]
        index1 = index1[0]
        
        index = np.where(t>timeWindow[1])
        index2 = index[0]
        index2 = index2[0]
        
        t_window = t[index1:index2]
        
        evokedAfterITD = evoked[index1:index2]
        evoked_fft = np.fft.fft(evokedAfterITD)
        freq = np.linspace(0, fs, len(t_window))
        pl.figure()
        pl.plot(freq, np.abs(evoked_fft.real))
        pl.show()
##################################################################################################################################################    
    if stimulus == 'ITD':
        timeWindow = [0.050, 0.25, 1.050, 1.25];
        epochs1, amp1, amp1Aud = evoked([1, 5], timeWindow)
        epochs1_left, amp1_left, amp1_leftAud = evoked([1], timeWindow)
        epochs1_right, amp1_right, amp1_rightAud = evoked([5], timeWindow)

        epochs2, amp2, amp2Aud = evoked([2, 6], timeWindow)
        epochs2_left, amp2_left, amp2_leftAud = evoked([2], timeWindow)
        epochs2_right, amp2_right, amp2_rightAud = evoked([6], timeWindow)
        
        epochs3, amp3, amp3Aud = evoked([3, 7], timeWindow)
        epochs3_left, amp3_left, amp3_leftAud = evoked([3], timeWindow)
        epochs3_right, amp3_right, amp3_rightAud = evoked([7], timeWindow)
        
        epochs4, amp4, amp4Aud = evoked([4, 8], timeWindow)
        epochs4_left, amp4_left, amp4_leftAud = evoked([4], timeWindow)
        epochs4_right, amp4_right, amp4_rightAud = evoked([8], timeWindow)
        #fft_evoked(evoked4, epochs4.times, 4096.0, [1.050, 1.5])
        # Average evoked response across conditions
        epochs_avg, amp_avg, amp_avgAud = evoked([1, 2, 3, 4, 5, 6, 7, 8], timeWindow)
        epochs_avg_left, amp_avg_left, amp_avg_leftAud = evoked([1, 2, 3, 4], timeWindow)
        epochs_avg_right, amp_avg_right, amp_avg_rightAud = evoked([5, 6, 7, 8], timeWindow)
        
        ampAllBoth = [amp1, amp2, amp3, amp4, amp_avg]
        ampAllLeft = [amp1_left, amp2_left, amp3_left, amp4_left, amp_avg_left]
        ampAllRight = [amp1_right, amp2_right, amp3_right, amp4_right, amp_avg_right]
        
        ampAudBoth = [amp1Aud, amp2Aud, amp3Aud, amp4Aud, amp_avgAud]
        ampAudLeft = [amp1_leftAud, amp2_leftAud, amp3_leftAud, amp4_leftAud, amp_avg_leftAud]
        ampAudRight = [amp1_rightAud, amp2_rightAud, amp3_rightAud, amp4_rightAud, amp_avg_rightAud]
        
#        pl.plot([amp1, amp2, amp3, amp4])
        
        t = epochs_avg.times
        freqs = np.arange(1., 50., 1.) # chaneg as needed
        itc_mean_all, itc_mean_auditory = itc(t, epochs_avg, 'avg')
        itc_mean_all_left, itc_mean_auditory_left = itc(t, epochs_avg_left, 'avg_left')
        itc_mean_all_right, itc_mean_auditory_right = itc(t, epochs_avg_right, 'avg_right')
        
        itc_mean_all_20us, itc_mean_auditory_20us = itc(t, epochs1, '20us')
        itc_mean_all_20us_left, itc_mean_auditory_20us_left = itc(t, epochs1_left, '20us_left')
        itc_mean_all_20us_right, itc_mean_auditory_20us_right = itc(t, epochs1_right, '20us_right')
        
        itc_mean_all_60us, itc_mean_auditory_60us =itc(t, epochs2, '60us')
        itc_mean_all_60us_left, itc_mean_auditory_60us_left = itc(t, epochs2_left, '60us_left')
        itc_mean_all_60us_right, itc_mean_auditory_60us_right = itc(t, epochs2_right, '60us_right')
        
        itc_mean_all_180us, itc_mean_auditory_180us = itc(t, epochs3, '180us')
        itc_mean_all_180us_left, itc_mean_auditory_180us_left = itc(t, epochs3_left, '180us_left')
        itc_mean_all_180us_right, itc_mean_auditory_180us_right = itc(t, epochs3_right, '180us_right')
        
        itc_mean_all_540us, itc_mean_auditory_540us = itc(t, epochs4, '540us')
        itc_mean_all_540us_left, itc_mean_auditory_540us_left = itc(t, epochs4_left, '540us_left')
        itc_mean_all_540us_right, itc_mean_auditory_540us_right = itc(t, epochs4_right, '540us_right')
        
        
#        itcMeanType = itc_mean_all
        cond = ''
        plot_spectrogram(itc_mean_auditory, t, freqs, cond)
#        itc_norm = itc_normalization(itcMeanType, freqs, t)
#        peak_all = plot_avg(itc_norm, t, cond)
        
        itcMeanType = [itc_mean_all_20us, itc_mean_all_60us, itc_mean_all_180us, itc_mean_all_540us, itc_mean_all]
        peaksAllBoth = [0]*len(itcMeanType)
        for i in range(len(itcMeanType)):
            itc_norm = itc_normalization(itcMeanType[i], freqs, t)
            peaksAllBoth[i] = peak(itc_norm, t)
            
        itcMeanType = [itc_mean_all_20us_left, itc_mean_all_60us_left, itc_mean_all_180us_left, itc_mean_all_540us_left, itc_mean_all_left]
        peaksAllLeft = [0]*len(itcMeanType)
        for i in range(len(itcMeanType)):
            itc_norm = itc_normalization(itcMeanType[i], freqs, t)
            peaksAllLeft[i] = peak(itc_norm, t)
            
        itcMeanType = [itc_mean_all_20us_right, itc_mean_all_60us_right, itc_mean_all_180us_right, itc_mean_all_540us_right, itc_mean_all_right]
        peaksAllRight = [0]*len(itcMeanType)
        for i in range(len(itcMeanType)):
            itc_norm = itc_normalization(itcMeanType[i], freqs, t)
            peaksAllRight[i] = peak(itc_norm, t)
            
        itcMeanType = [itc_mean_auditory_20us, itc_mean_auditory_60us, itc_mean_auditory_180us, itc_mean_auditory_540us, itc_mean_auditory]
        peaksAuditoryBoth = [0]*len(itcMeanType)
        for i in range(len(itcMeanType)):
            itc_norm = itc_normalization(itcMeanType[i], freqs, t)
            peaksAuditoryBoth[i] = peak(itc_norm, t)
            
        itcMeanType = [itc_mean_auditory_20us_left, itc_mean_auditory_60us_left, itc_mean_auditory_180us_left, itc_mean_auditory_540us_left, itc_mean_auditory_left]
        peaksAuditoryLeft = [0]*len(itcMeanType)
        for i in range(len(itcMeanType)):
            itc_norm = itc_normalization(itcMeanType[i], freqs, t)
            peaksAuditoryLeft[i] = peak(itc_norm, t)
            
        itcMeanType = [itc_mean_auditory_20us_right, itc_mean_auditory_60us_right, itc_mean_auditory_180us_right, itc_mean_auditory_540us_right, itc_mean_auditory_right]
        peaksAuditoryRight = [0]*len(itcMeanType)
        for i in range(len(itcMeanType)):
            itc_norm = itc_normalization(itcMeanType[i], freqs, t)
            peaksAuditoryRight[i] = peak(itc_norm, t)
        
    
    elif stimulus == 'Atten':
        # Average evoked response across short-stream conditions
        epochs_short = mne.Epochs(raw, eves2, [3, 4, 7, 8], tmin = -0.5, proj = True, tmax = 4.2, 
                            baseline = (-0.5, 0.), reject = dict(eeg=150e-6)) # change the channels as needed
        t_short = epochs_short.times
        evoked_shortStream = epochs_short.average() 
        
        # Average evoked response across long-stream conditions
        epochs_long = mne.Epochs(raw, eves2, [1, 2, 5, 6], tmin = -0.5, proj = True, tmax = 4.2, 
                            baseline = (-0.5, 0), reject = dict(eeg=150e-6))
        t_long = epochs_long.times
        evoked_longStream = epochs_long.average()
        
        # Average visual evoked response
        epochs_cue = mne.Epochs(raw, eves2, [9, 10, 11, 12, 13, 14, 15, 16],
                                tmin = -0.5, proj = True, tmax = 1.0, 
                                baseline = (-0.5, 0), reject = dict(eeg=150e-6))
        t_cue = epochs_cue.times
        evoked_cue = epochs_cue.average()
        
        itc_short_all, itc_short_auditory = itc (t_short, epochs_short, 'short stream')
        itc_long_all, itc_long_auditory = itc (t_long, epochs_long, 'long stream')
        itc_cue_all, itc_cue_auditory = itc (t_cue, epochs_cue, 'visual cues')
        
        cond = 'long_stream'
        freqs = np.arange(1., 50., 1.) # chaneg as needed
        t = t_long
        itcType = itc_long_auditory
        itc_norm = itc_normalization(itcType, freqs, t)
        peak_all = plot_avg(itc_norm, t, cond)
        
    elif stimulus == 'FM':
        epochs_positive = mne.Epochs(raw, eves2, [1], tmin = -0.05, proj = True, tmax = 0.15, 
                            baseline = (-0.05, 0.), reject = dict(eeg=150e-6)) # change the channels as needed
        t = epochs_positive.times
        evoked_positive = epochs_positive.average() 
        evoked_grand = auditoryAvg(evoked_positive)
        evoked_fft = np.fft.fft(evoked_grand)
        fs = 16384
        freq = np.linspace(0, fs, len(evoked_fft))
        pl.figure()
        pl.plot(freq, np.abs(evoked_fft))
        pl.show()
        
        
    #    code for trying to figure out the right frequency cutt-off for averaging    
    #    evokedPeaks = []
    #    firstPeaks = []
    #    firstPeak_sums = []
    #    evokedPeaksNoNorm = []
    #    freqRange = range(10, 35)
    #    for k, freq in enumerate(freqRange):
    #        freqSub = np.where(freqs<freq) # CHANGE AS NEEDED FOR DIFFERENT SUBJECTS, refer to time-freq plot of itc
    #        itc_ave_below22Hz = itc_data_mean[freqSub[0], :].mean(axis=0)
    #        itc_sum = np.sum(itc_data_mean[freqSub[0],:], axis = 0)
    #        index = np.where(t>0)
    #        index1 = index[0]
    #        index1 = index1[0]
    #        index = np.where(t<0.3) # CHANGE AS NEEDED FOR DIFFERENT SUBJECTS
    #        index2 = index[0]
    #        index2 = index2[-1]
    #        
    #        noiseFloor = itc_ave_below22Hz[0:index1].mean(axis=0)
    #        itc_ave_below22Hz = itc_ave_below22Hz - noiseFloor
    #        
    #        firstPeakAmp = np.max(itc_ave_below22Hz[index1:index2])
    #        firstPeakAmp_sum = np.max(itc_sum[index1:index2])
    #        firstPeaks += [firstPeakAmp,]
    #        firstPeak_sums += [firstPeakAmp_sum,]
    #        itc_norm_below22Hz = itc_ave_below22Hz/firstPeakAmp
    #        
    #        # storing the peak amplitude of ITD evoked response
    #        index = np.where(t>0.98)
    #        index1 = index[0]
    #        index1 = index1[0]
    #        index = np.where(t<1.5) # CHANGE AS NEEDED FOR DIFFERENT SUBJECTS
    #        index2 = index[0]
    #        index2 = index2[-1]
    #        evokedPeakAmp = np.max(itc_norm_below22Hz[index1:index2])
    #        evokedPeakAmp_noNorm = np.max(itc_ave_below22Hz[index1:index2])
    #        evokedPeaks += [evokedPeakAmp,]
    #        evokedPeaksNoNorm += [evokedPeakAmp_noNorm]
    #    ratio = np.divide(firstPeaks,evokedPeaksNoNorm)
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