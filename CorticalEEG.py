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
OS = 'Mac'

if stimulus == 'ITD':
    if OS == 'Ubuntu':
        froot = '/media/agudemu/Storage/Data/EEG/ITD'
    else:
        froot = '/Users/baoagudemu1/Desktop/Lab/Experiment/DataAnalysis/Data'
    subjlist = ['S078']     
else:
    if OS == 'Ubuntu':
        froot = '/media/agudemu/Storage/Data/EEG/Atten'
    else:
        froot = '/Users/baoagudemu1/Desktop/Lab/EEG-Python/Atten'
    subjlist = ['S011']     
#subjlist = ['S025', 'S028', 'S031', 'S043', 'S046', 'S072', 'S075', 'S078', 'S083', 'S117', 'S119', 'S123', 'S127', 'S128', 'S132', 
# S133, S139, S140, S143, S144, S145, S149]
# S084 and S135 need different projecions


for subj in subjlist:
    fpath = froot + '/' + subj + '/'
    print 'Running subject', subj
    
    # load data and read event channels
    rawlist = []
    evelist = []
    # extracting bdf filenames into bdfs
    if stimulus == 'ITD':
        bdfs = fnmatch.filter(os.listdir(fpath), subj + '_ITD*.bdf') 
    else:
        bdfs = fnmatch.filter(os.listdir(fpath), subj + '_Atten*.bdf') 
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
    raw.filter(l_freq = 0.5, h_freq=50) # if channels are noisy, adjust the filter parameters
    # raw.plot(events=eves2)
    
    # SSP for blinks
    blinks = find_blinks(raw, ch_name = ['A1',], l_trans_bandwidth=0.5,
                         l_freq=1.0) # A1 is the closest electrode to eyebrow
    # blink and eves2 triggers can be combined using np.concatenate((eves2, blinks), axis = 0)
    # raw.plot(events=blinks) shows the lines at eye blinks
    
    # the trigger for blinks can be chosen to be starting from 1000, just to make sure it doesn't collide with the triggers for conditions
    epochs_blinks = mne.Epochs(raw, blinks, 998, tmin = -0.5, tmax = 0.5, 
                               proj = False, baseline = (-0.5, 0), 
                               reject=dict(eeg=500e-6)) 
    evoked_blinks = epochs_blinks.average()
    evoked_blinks_data = evoked_blinks.data[np.arange(32),:]
    # PCA is only applied to the epochs around eye blinks. Since the eye blinks are 
    # contributing the most to the variance within this chunk of window, 
    # the first PCA (first eigenvector) is going to be due to the eye blink 
    # for sure and removed. If the PCA was performed on the whole samples, we wouldn't
    # know which vector is going to be linked to the eye blink
    # for "n_eeg", it's recommended to remove only the biggest projection, which is eye blinks in this case
    # greater n_eeg removes more nueral data, which might not be favorable
    n_eeg = 4
    blink_projs = compute_proj_epochs(epochs_blinks, n_grad=0,
                                      n_mag=0, n_eeg=n_eeg,
                                      verbose='DEBUG') 
    # time course of the PCA components
#        blink_vectors = np.zeros(shape = (n_eeg,n_eeg))
#        for k in np.arange(n_eeg):
#            blink_value = blink_projs[k].values()
#            blink_data = blink_value[2] # dictionary object containing data (key)
#            blink_vectors[k] = blink_data['data']
#        blink_vectors = np.transpose(blink_vectors)
#        projs_timeCourse = np.matmul(blink_vectors, evoked_blinks_data)
#        
#        for k in np.arange(n_eeg):
#            pl.subplot(n_eeg,1,k+1)
#            pl.plot(projs_timeCourse[k,:])
        
    #raw.add_proj(blink_projs) # adding all projections
    raw.add_proj([blink_projs[0], blink_projs[2]]) # raw.del_proj()
    #raw.add_proj(blink_projs)
    
    # raw.plot_projs_topomap() shows the 4 max PCAs (eigenvectors)
    
    # if channels are too noisy, play with n_eeg, if the second variance is acting more than
    # the first, that means the channels are contaminated not just by the eye blinks, but also
    # from other sources, raw.plot(events = blinks, show_options = True) could show the options for applying different projections

    # MANUALLY SELECT PROJECTIONS BY PLOTTING raw.plot_projs_topomap
    # REMOVE EXTRA PROJS USING raw.del_proj -- Remember index starts at 0
###########################################################################################################################################
    def noiseFloorEstimate (t, evoked):
        index = np.where(t>0)
        index1 = index[0]
        index1 = index1[0]
        index = np.where(t<0.2) # CHANGE AS NEEDED FOR DIFFERENT SUBJECTS, 0.3
        index2 = index[0]
        index2 = index2[-1]
        
        noiseFloor = evoked[0:index1].mean(axis=0)
        return(noiseFloor, index1, index2)
###########################################################################################################################################    
    
    def itc (t, epochs, cond):
        # computation of inter-trial-coherence (itc)
        freqs = np.arange(5., 100., 2.) # CHANGE AS NEEDED
        n_cycles = freqs/4. # time resolution is 0.25 s (1 s has "freq" number of cycles)
        time_bandwidth = 2.0 # number of taper = time_bandwidth product - 1 
        # usually n_cycles and time_bandwidth are fixed, which determines the frequency resolution 
        power, itc = tfr_multitaper(epochs, freqs = freqs,  n_cycles = n_cycles,
                   time_bandwidth = time_bandwidth, return_itc = True, n_jobs = 4)
        # itc.plot([channel number], mode = 'mean')
        # np.savez(fpath+'/'+'itc_power', itc = itc.data, power = power.data) # saves multiple arrays, saved in .npz format
        # npzFile = np.load(fpath+'/'+'itc_power.npz'), npzFiles.files, npzFile['itc']
        # np.save(fpath+'/'+'itc_power', itc) # only one array, saved in .npy format
        # itc = itc seems to save the attributes of the itc instead of the real data
        itc_copy = itc.copy()
        # itc_copy.plot_topo(baseline = (-0.5, 0), mode = 'zscore') 
        itc_data = itc_copy.data
        np.savez(froot+'/'+'itcs' + '/'+'itc'+str(cond)+'_'+subj, itc = itc_data, t = t, freqs = freqs); # itc_data is the most time consuming variable
        # averaging across channels
        itc_data_mean = itc_data.mean(axis = 0) # CHANGE AS NEEDED: average across all channels 
        #itc_data_mean = itc_data[[4, 26, 25, 30, 31], :, :].mean(axis = 0) # these channels are auditory 
        # if bad channels were added manually, select good channels
        power_copy = power.copy()
        power_data = power_copy.data
        np.savez(froot+'/'+'powers' + '/'+'power'+str(cond)+'_'+subj, power = power_data, t = t, freqs = freqs);
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
        #   pl.plot(t, itc_data_mean[0:8, :].mean(axis=0))
        freqSub = np.where(freqs<10) # CHANGE AS NEEDED FOR DIFFERENT SUBJECTS, refer to time-freq plot of itc, 23
        itc_ave = itc_data_mean[freqSub[0], :].mean(axis=0)
        noiseFloor, index1, index2 = noiseFloorEstimate(t, itc_ave)
        itc_ave = itc_ave - noiseFloor
        
        firstPeakAmp = np.absolute(np.max(itc_ave[index1:index2])) # ABSOLUTE? POLARITY MATTERS
        itc_norm = itc_ave/firstPeakAmp
        return(itc_norm) 
        
    def plot (t, itc_norm, cond):
        pl.figure()
        
        pl.plot(t, itc_norm)
        pl.xlabel('Time (s)', fontsize=14)
        pl.ylabel('Normalized response', fontsize=14)
        pl.title('Phase locking:'+str(cond), fontsize=14)
        ax = pl.gca()
        ax.tick_params(labelsize=14)
        pl.show()
        return
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
        
        return(evokedAmpNor)
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
        evoked = epochs.average()
        # noiseFloor, _, _, = noiseFloorEstimate(t, evoked.data[31]) # noise floor is very small, since DC has been filtered out
        # evoked_chann32 = evoked.data[31] - noiseFloor
        evoked.plot(picks = [31])
        evokedAud = auditoryAvg(evoked)
        evokedAud = butter_lowpass_filter(evokedAud, 40.8, 44100, 5)
        pl.plot(t*1e3, evokedAud*1e6)
        amp = normalizedN1p2(evokedAud, t, timeWindow)
        return(amp, epochs)
##################################################################################################################################################        
    if stimulus == 'ITD':
        amp1, epochs1 = evoked([1, 5], [0.075, 0.25, 1.075, 1.3])
        
        amp2, epochs2 = evoked([2, 6], [0.075, 0.25, 1, 1.3])
        
        amp3, epochs3 = evoked([3, 7], [0.075, 0.25, 1, 1.3])
        
        amp4, epochs4 = evoked([4, 8], [0.075, 0.25, 1, 1.3])
        # Average evoked response across conditions
        amp, epochs = evoked([1, 2, 3, 4, 5, 6, 7, 8], [0.075, 0.25, 1, 1.3])
        # epochs.save(fpath+'/'+'no_blinks_epo.fif', split_size='2GB') # saving epochs into .fif format
        
        pl.plot([amp1, amp2, amp3, amp4])
        
        t = epochs.times
        itc_avg = itc (t, epochs, 'avg')
        plot(t, itc_avg, 'avg')
        itc_1 = itc (t, epochs1, '20us')
        plot(t, itc_1, '20us')
        itc_2 = itc (t, epochs2, '60us')
        plot(t, itc_2, '60us')
        itc_3 = itc (t, epochs3, '180us')
        plot(t, itc_3, '180us')
        itc_4 = itc (t, epochs4, '540us')
        plot(t, itc_4, '540us')
    
    elif stimulus == 'Atten':
        # Average evoked response across short-stream conditions
        epochs_short = mne.Epochs(raw, eves2, [3, 4, 7, 8], tmin = -0.5, proj = True, tmax = 4.2, 
                            baseline = (-0.5, 0.), reject = dict(eeg=150e-6)) # change the channels as needed
        t = epochs_short.times
        evoked_shortStream = epochs_short.average() 
        
        # Average evoked response across long-stream conditions
        epochs_long = mne.Epochs(raw, eves2, [1, 2, 5, 6], tmin = -0.5, proj = True, tmax = 4.2, 
                            baseline = (-0.5, 0), reject = dict(eeg=150e-6))
        evoked_longStream = epochs_long.average()
        
        # Average visual evoked response
        epochs_cue = mne.Epochs(raw, eves2, [9, 10, 11, 12, 13, 14, 15, 16],
                                tmin = -0.5, proj = True, tmax = 1.0, 
                                baseline = (-0.5, 0), reject = dict(eeg=150e-6))
        evoked_cue = epochs_cue.average()
        
        itc_short = itc (t, epochs_short, 'short stream')
        plot(t, itc_short, 'short stream')
        itc_long = itc (t, epochs_long, 'long stream')
        plot(t, itc_long, 'long stream')
        itc_cue = itc (t, epochs_cue, 'visual cues')
        plot(t, itc_cue, 'visual cue')
    
    
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