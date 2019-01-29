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
import xlwt


def noiseFloorEstimate(t, evoked):
    index = np.where(t>0)
    index1 = index[0]
    index1 = index1[0]
    index = np.where(t<0.3) # CHANGE AS NEEDED FOR DIFFERENT SUBJECTS, 0.3
    index2 = index[0]
    index2 = index2[-1]
    
    noiseFloor = evoked[0:index1].mean(axis=0)
    return noiseFloor, index1, index2

def itc_normalization(itc, freqs, t):
    if stimulus == 'ITD':
        freqSub = np.where(freqs<20)
    elif stimulus == 'Atten':
        freqSub = np.where(freqs<50) 
        
    itc_ave = itc[freqSub[0], :].mean(axis=0)
    noiseFloor, index1, index2 = noiseFloorEstimate(t, itc_ave)
    itc_ave = itc_ave - noiseFloor
    
    firstPeakAmp = np.absolute(np.max(itc_ave[index1:index2])) 
    itc_norm = itc_ave/firstPeakAmp
    return itc_norm

def peak(itc_norm, t):
    index = np.where(t>0.9)
    index1 = index[0]
    index1 = index1[0]
    index = np.where(t<1.3) # CHANGE AS NEEDED FOR DIFFERENT SUBJECTS, 0.3
    index2 = index[0]
    index2 = index2[-1]
    return round(max(itc_norm[index1:index2]), 2)

def itc(t, epochs, cond):
    # computation of inter-trial-coherence (itc)
    freqs = np.arange(1., 50., 1.) # CHANGE AS NEEDED
    n_cycles = freqs/4. # time resolution is 0.25 s (1 s has "freq" number of cycles)
    time_bandwidth = 2.0 # number of taper = time_bandwidth product - 1 
    # usually n_cycles and time_bandwidth are fixed, which determines the frequency resolution 
    power, itc = tfr_multitaper(epochs, freqs = freqs,  n_cycles = n_cycles,
               time_bandwidth = time_bandwidth, return_itc = True, n_jobs = 4)
    # itc.plot([channel number], mode = 'mean')
    # itc.plot_topo(baseline = (-0.5, 0), mode = 'zscore') 
    itc_copy = itc.copy()
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
            
def auditoryAvg(evoked):
    evokedAvg = np.zeros(shape = (1, len(evoked.data[0])))
    audChanns = [4, 26, 25, 30, 31]
    for i in audChanns:
        evokedAvg = evokedAvg + evoked.data[i]
    evokedAvg = evokedAvg / len(audChanns)
    return evokedAvg[0]
   
def evoked(raw, eves2, triggerID, timeWindow):
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
    return epochs, round(ampAll, 2), round(ampAud, 2)
        
def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order)
    y = lfilter(b, a, data)
    return y
   
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

def processing(stimulus, subj):
    # load data and read event channels
    rawlist = []
    evelist = []
    # extracting bdf filenames into bdfs
    if stimulus == 'ITD':
        bdfs = fnmatch.filter(os.listdir(fpath), subj + '_ITD*.bdf') 
    elif stimulus == 'Atten':
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
    if stimulus == 'ITD' or stimulus == 'Atten':
        raw.filter(l_freq = 0.5, h_freq=50) # if channels are noisy, adjust the filter parameters
    # raw.plot(events=eves2)
    # SSP for blinks
    blinks = find_blinks(raw, ch_name = ['A1',], l_trans_bandwidth=0.5,
                         l_freq=1.0) # A1 is the closest electrode to eyebrow
    # blink and eves2 triggers can be combined using np.concatenate((eves2, blinks), axis = 0)
    # raw.plot(events=blinks) shows the lines at eye blinks
    
    # the trigger for blinks can be chosen to be starting from 1000, just to make sure it doesn't collide with the triggers for conditions
    epochs_blinks = mne.Epochs(raw, blinks, 998, tmin = -0.25, tmax = 0.25, 
                               proj = False, baseline = (-0.25, 0), 
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
    projN = np.zeros(2)
    if subj in ['S025', 'S031', 'S046', 'S117', 'S123', 'S127', 'S128', 'S132', 'S133', 'S143', 'S149', 'S051', 'S183', 'S185', 'S187', 'S189', 'S193', 'S195', 'S196', 'S043', 'S072', 'S075', 'S078', 'S191']:
        #raw.add_proj(blink_projs) # adding all projections
        raw.add_proj([blink_projs[0], blink_projs[2]]) # raw.del_proj()   
        projN = [1, 3]
    elif subj in ['S135', 'S192', 'S194', 'S197', 'S199']:
        raw.add_proj([blink_projs[0], blink_projs[1]]) # raw.del_proj()   
        projN = [1, 2]
    elif subj in ['S084']:
        raw.add_proj([blink_projs[0]]) # raw.del_proj()   
        projN = [1, 0]
    
    # raw.plot_projs_topomap() shows the 4 max PCAs (eigenvectors)
    # raw.plot(events = blinks, show_options = True) could show the options for applying different projections
    
    # if channels are too noisy, play with n_eeg, if the second variance is acting more than
    # the first, that means the channels are contaminated not just by the eye blinks, but also
    # from other sources

    # MANUALLY SELECT PROJECTIONS BY PLOTTING raw.plot_projs_topomap
    # REMOVE EXTRA PROJS USING raw.del_proj -- Remember index starts at 0
    return raw, eves2, projN

def dataEvoked(raw, eves, ids, timeWindow):
    epochs, amp, ampAud = evoked(raw, eves, ids, timeWindow)
    left_index = range(len(ids)/2)
    epochs_left, amp_left, amp_leftAud = evoked(raw, eves, ids[0:len(ids)/2], timeWindow)
    right_index = range(left_index[-1]+1, ids[-1])
    epochs_right, amp_right, amp_rightAud = evoked(raw, eves, ids[len(ids)/2:len(ids)], timeWindow)    
    return epochs, amp, ampAud, epochs_left, amp_left, amp_leftAud, epochs_right, amp_right, amp_rightAud

def dataITC(t, epochs, epochs_left, epochs_right, cond1, cond2, cond3):
    itc_all, itc_auditory = itc(t, epochs, cond1)
    itc_all_left, itc_auditory_left = itc(t, epochs_left, cond2)
    itc_all_right, itc_auditory_right = itc(t, epochs_right, cond3)
    return itc_all, itc_auditory, itc_all_left, itc_auditory_left, itc_all_right, itc_auditory_right

def peakITC(itcMeanType, freqs, t):
    peaks = [0]*len(itcMeanType)
    for i in range(len(itcMeanType)):
        itc_norm = itc_normalization(itcMeanType[i], freqs, t)
        peaks[i] = peak(itc_norm, t)
    return peaks
##############################################################################################################################################################

stimulus = 'ITD'
OS = 'Ubuntu'

#subjlist = ['S025', 'S031', 'S046', 'S117', 'S123', 'S127', 'S128', 'S132', 'S133', 'S143', 'S149', 'S051', 'S183', 'S185', 'S187', 'S189', 'S193', 'S195', 'S196', 'S043', 'S072', 'S075', 'S078', 'S135', 'S192', 'S194', 'S197', 'S199']     
subjlist = ['S191', 'S084']
if stimulus == 'ITD':
    if OS == 'Ubuntu':
        froot = '/media/agudemu/Storage/Data/EEG/ITD'
    else:
        froot = '/Users/baoagudemu1/Desktop/Lab/Experiment/DataAnalysis/Data'    
    wb = xlwt.Workbook()
    sheet_itc_all = wb.add_sheet('itc_all_channels')
    sheet_itc_aud = wb.add_sheet('itc_auditory_channels')
    sheet_amp_all = wb.add_sheet('amplitude_all_channels')
    sheet_amp_aud = wb.add_sheet('amplitude_auditory_channels')
    sheet_num_trials = wb.add_sheet('number of trials')
    style_bold = xlwt.easyxf('font: bold 1')
    
    sheet_num_trials.write(1, 0, 'Subject ID', style_bold)
    sheet_num_trials.write(0, 2, 'Num of trials', style_bold)
    sheet_num_trials.write(1, 1, 'Projections', style_bold)
    sheet_num_trials.write(1, 2, '20us_left', style_bold)
    sheet_num_trials.write(1, 3, '60us_left', style_bold)
    sheet_num_trials.write(1, 4, '180us_left', style_bold)
    sheet_num_trials.write(1, 5, '540us_left', style_bold)
    sheet_num_trials.write(1, 6, '20us_right', style_bold)
    sheet_num_trials.write(1, 7, '60us_right', style_bold)
    sheet_num_trials.write(1, 8, '180us_right', style_bold)
    sheet_num_trials.write(1, 9, '540us_right', style_bold)
    sheet_num_trials.write(1, 10, 'ASSR', style_bold)
    
    sheet_itc_all.write(1, 0, 'Subject ID', style_bold)
    sheet_itc_all.write(0, 1, 'Left', style_bold) 
    sheet_itc_all.write(0, 7, 'Right', style_bold) 
    sheet_itc_all.write(0, 13, 'Both', style_bold) 
    sheet_itc_all.write(1, 1, '20 us', style_bold) 
    sheet_itc_all.write(1, 2, '60 us', style_bold)
    sheet_itc_all.write(1, 3, '180 us', style_bold)
    sheet_itc_all.write(1, 4, '540 us', style_bold)
    sheet_itc_all.write(1, 5, 'Avg', style_bold)
    
    sheet_itc_all.write(1, 7, '20 us', style_bold) 
    sheet_itc_all.write(1, 8, '60 us', style_bold)
    sheet_itc_all.write(1, 9, '180 us', style_bold)
    sheet_itc_all.write(1, 10, '540 us', style_bold)
    sheet_itc_all.write(1, 11, 'Avg', style_bold)
    
    sheet_itc_all.write(1, 13, '20 us', style_bold) 
    sheet_itc_all.write(1, 14, '60 us', style_bold)
    sheet_itc_all.write(1, 15, '180 us', style_bold)
    sheet_itc_all.write(1, 16, '540 us', style_bold)
    sheet_itc_all.write(1, 17, 'Avg', style_bold)
    
    sheet_itc_aud.write(1, 0, 'Subject ID', style_bold)
    sheet_itc_aud.write(0, 1, 'Left', style_bold) 
    sheet_itc_aud.write(0, 7, 'Right', style_bold) 
    sheet_itc_aud.write(0, 13, 'Both', style_bold) 
    sheet_itc_aud.write(1, 1, '20 us', style_bold) 
    sheet_itc_aud.write(1, 2, '60 us', style_bold)
    sheet_itc_aud.write(1, 3, '180 us', style_bold)
    sheet_itc_aud.write(1, 4, '540 us', style_bold)
    sheet_itc_aud.write(1, 5, 'Avg', style_bold)
    
    sheet_itc_aud.write(1, 7, '20 us', style_bold) 
    sheet_itc_aud.write(1, 8, '60 us', style_bold)
    sheet_itc_aud.write(1, 9, '180 us', style_bold)
    sheet_itc_aud.write(1, 10, '540 us', style_bold)
    sheet_itc_aud.write(1, 11, 'Avg', style_bold)
    
    sheet_itc_aud.write(1, 13, '20 us', style_bold) 
    sheet_itc_aud.write(1, 14, '60 us', style_bold)
    sheet_itc_aud.write(1, 15, '180 us', style_bold)
    sheet_itc_aud.write(1, 16, '540 us', style_bold)
    sheet_itc_aud.write(1, 17, 'Avg', style_bold)
    
    sheet_amp_all.write(1, 0, 'Subject ID', style_bold)
    sheet_amp_all.write(0, 1, 'Left', style_bold) 
    sheet_amp_all.write(0, 7, 'Right', style_bold) 
    sheet_amp_all.write(0, 13, 'Both', style_bold) 
    sheet_amp_all.write(1, 1, '20 us', style_bold) 
    sheet_amp_all.write(1, 2, '60 us', style_bold)
    sheet_amp_all.write(1, 3, '180 us', style_bold)
    sheet_amp_all.write(1, 4, '540 us', style_bold)
    sheet_amp_all.write(1, 5, 'Avg', style_bold)
    
    sheet_amp_all.write(1, 7, '20 us', style_bold) 
    sheet_amp_all.write(1, 8, '60 us', style_bold)
    sheet_amp_all.write(1, 9, '180 us', style_bold)
    sheet_amp_all.write(1, 10, '540 us', style_bold)
    sheet_amp_all.write(1, 11, 'Avg', style_bold)
    
    sheet_amp_all.write(1, 13, '20 us', style_bold) 
    sheet_amp_all.write(1, 14, '60 us', style_bold)
    sheet_amp_all.write(1, 15, '180 us', style_bold)
    sheet_amp_all.write(1, 16, '540 us', style_bold)
    sheet_amp_all.write(1, 17, 'Avg', style_bold)
    
    sheet_amp_aud.write(1, 0, 'Subject ID', style_bold)
    sheet_amp_aud.write(0, 1, 'Left', style_bold) 
    sheet_amp_aud.write(0, 7, 'Right', style_bold) 
    sheet_amp_aud.write(0, 13, 'Both', style_bold) 
    sheet_amp_aud.write(1, 1, '20 us', style_bold) 
    sheet_amp_aud.write(1, 2, '60 us', style_bold)
    sheet_amp_aud.write(1, 3, '180 us', style_bold)
    sheet_amp_aud.write(1, 4, '540 us', style_bold)
    sheet_amp_aud.write(1, 5, 'Avg', style_bold)
    
    sheet_amp_aud.write(1, 7, '20 us', style_bold) 
    sheet_amp_aud.write(1, 8, '60 us', style_bold)
    sheet_amp_aud.write(1, 9, '180 us', style_bold)
    sheet_amp_aud.write(1, 10, '540 us', style_bold)
    sheet_amp_aud.write(1, 11, 'Avg', style_bold)
    
    sheet_amp_aud.write(1, 13, '20 us', style_bold) 
    sheet_amp_aud.write(1, 14, '60 us', style_bold)
    sheet_amp_aud.write(1, 15, '180 us', style_bold)
    sheet_amp_aud.write(1, 16, '540 us', style_bold)
    sheet_amp_aud.write(1, 17, 'Avg', style_bold)
    wb.save(froot+'/EEGdata.xls')
elif stimulus == 'Atten':
    if OS == 'Ubuntu':
        froot = '/media/agudemu/Storage/Data/EEG/Atten'
    else:
        froot = '/Users/baoagudemu1/Desktop/Lab/EEG-Python/Atten'

subjN = 0;
for subj in subjlist:
    subjN = subjN + 1;
    fpath = froot + '/' + subj + '/'
    print 'Running subject', subj
    
    [raw, eves2, projN] = processing(stimulus, subj)
    
    if stimulus == 'ITD':
        timeWindow = [0.050, 0.25, 1.050, 1.25];
        
        epochs1, amp1, amp1Aud, epochs1_left, amp1_left, amp1_leftAud, epochs1_right, amp1_right, amp1_rightAud = dataEvoked(raw, eves2, [1, 5], timeWindow)
        
        epochs2, amp2, amp2Aud, epochs2_left, amp2_left, amp2_leftAud, epochs2_right, amp2_right, amp2_rightAud = dataEvoked(raw, eves2, [2, 6], timeWindow)

        epochs3, amp3, amp3Aud, epochs3_left, amp3_left, amp3_leftAud, epochs3_right, amp3_right, amp3_rightAud = dataEvoked(raw, eves2, [3, 7], timeWindow)
        
        epochs4, amp4, amp4Aud, epochs4_left, amp4_left, amp4_leftAud, epochs4_right, amp4_right, amp4_rightAud = dataEvoked(raw, eves2, [4, 8], timeWindow)
        
        epochs_avg, amp_avg, amp_avgAud, epochs_avg_left, amp_avg_left, amp_avg_leftAud, epochs_avg_right, amp_avg_right, amp_avg_rightAud = dataEvoked(raw, eves2, [1, 2, 3, 4, 5, 6, 7, 8], timeWindow)
        
        ampAllBoth = [amp1, amp2, amp3, amp4, amp_avg]
        ampAllLeft = [amp1_left, amp2_left, amp3_left, amp4_left, amp_avg_left]
        ampAllRight = [amp1_right, amp2_right, amp3_right, amp4_right, amp_avg_right]
        
        ampAudBoth = [amp1Aud, amp2Aud, amp3Aud, amp4Aud, amp_avgAud]
        ampAudLeft = [amp1_leftAud, amp2_leftAud, amp3_leftAud, amp4_leftAud, amp_avg_leftAud]
        ampAudRight = [amp1_rightAud, amp2_rightAud, amp3_rightAud, amp4_rightAud, amp_avg_rightAud]
        #fft_evoked(evoked4, epochs4.times, 4096.0, [1.050, 1.5])
        # Average evoked response across conditions
        
        t = epochs_avg.times
        freqs = np.arange(1., 50., 1.) # chaneg as needed
        
        itc_mean_all, itc_mean_auditory, itc_mean_all_left, itc_mean_auditory_left, itc_mean_all_right, itc_mean_auditory_right = dataITC(t, epochs_avg, epochs_avg_left, epochs_avg_right, 'avg', 'avg_left', 'avg_right')
        
        itc_mean_all_20us, itc_mean_auditory_20us, itc_mean_all_20us_left, itc_mean_auditory_20us_left, itc_mean_all_20us_right, itc_mean_auditory_20us_right = dataITC(t, epochs1, epochs1_left, epochs1_right, '20us', '20us_left', '20us_right')
        
        itc_mean_all_60us, itc_mean_auditory_60us, itc_mean_all_60us_left, itc_mean_auditory_60us_left, itc_mean_all_60us_right, itc_mean_auditory_60us_right = dataITC(t, epochs2, epochs2_left, epochs2_right, '60us', '60us_left', '60us_right')
        
        itc_mean_all_180us, itc_mean_auditory_180us, itc_mean_all_180us_left, itc_mean_auditory_180us_left, itc_mean_all_180us_right, itc_mean_auditory_180us_right = dataITC(t, epochs3, epochs3_left, epochs3_right, '180us', '180us_left', '180us_right')
        
        itc_mean_all_540us, itc_mean_auditory_540us, itc_mean_all_540us_left, itc_mean_auditory_540us_left, itc_mean_all_540us_right, itc_mean_auditory_540us_right = dataITC(t, epochs4, epochs4_left, epochs4_right, '540us', '540us_left', '540us_right')
        
        
#        itcMeanType = itc_mean_all
        cond = ''
        plot_spectrogram(itc_mean_auditory, t, freqs, cond)
#        itc_norm = itc_normalization(itcMeanType, freqs, t)
#        peak_all = plot_avg(itc_norm, t, cond)
        
        itcMeanType = [itc_mean_all_20us, itc_mean_all_60us, itc_mean_all_180us, itc_mean_all_540us, itc_mean_all]
        peaksAllBoth = peakITC(itcMeanType, freqs, t)
       
        itcMeanType = [itc_mean_all_20us_left, itc_mean_all_60us_left, itc_mean_all_180us_left, itc_mean_all_540us_left, itc_mean_all_left]
        peaksAllLeft = peakITC(itcMeanType, freqs, t)
            
        itcMeanType = [itc_mean_all_20us_right, itc_mean_all_60us_right, itc_mean_all_180us_right, itc_mean_all_540us_right, itc_mean_all_right]
        peaksAllRight = peakITC(itcMeanType, freqs, t)
            
        itcMeanType = [itc_mean_auditory_20us, itc_mean_auditory_60us, itc_mean_auditory_180us, itc_mean_auditory_540us, itc_mean_auditory]
        peaksAuditoryBoth = peakITC(itcMeanType, freqs, t)
            
        itcMeanType = [itc_mean_auditory_20us_left, itc_mean_auditory_60us_left, itc_mean_auditory_180us_left, itc_mean_auditory_540us_left, itc_mean_auditory_left]
        peaksAuditoryLeft = peakITC(itcMeanType, freqs, t)
            
        itcMeanType = [itc_mean_auditory_20us_right, itc_mean_auditory_60us_right, itc_mean_auditory_180us_right, itc_mean_auditory_540us_right, itc_mean_auditory_right]
        peaksAuditoryRight = peakITC(itcMeanType, freqs, t)
        
        # storing data to spreadsheet
        sheet_itc_all.write(1+subjN, 0, subj) 
        sheet_itc_all.write(1+subjN, 1, str(peaksAllLeft[0])) 
        sheet_itc_all.write(1+subjN, 2, str(peaksAllLeft[1]))
        sheet_itc_all.write(1+subjN, 3, str(peaksAllLeft[2]))
        sheet_itc_all.write(1+subjN, 4, str(peaksAllLeft[3]))
        sheet_itc_all.write(1+subjN, 5, str(peaksAllLeft[4]))
        sheet_itc_all.write(1+subjN, 7, str(peaksAllRight[0])) 
        sheet_itc_all.write(1+subjN, 8, str(peaksAllRight[1]))
        sheet_itc_all.write(1+subjN, 9, str(peaksAllRight[2]))
        sheet_itc_all.write(1+subjN, 10, str(peaksAllRight[3]))
        sheet_itc_all.write(1+subjN, 11, str(peaksAllRight[4]))
        sheet_itc_all.write(1+subjN, 13, str(peaksAllBoth[0])) 
        sheet_itc_all.write(1+subjN, 14, str(peaksAllBoth[1]))
        sheet_itc_all.write(1+subjN, 15, str(peaksAllBoth[2]))
        sheet_itc_all.write(1+subjN, 16, str(peaksAllBoth[3]))
        sheet_itc_all.write(1+subjN, 17, str(peaksAllBoth[4])) 
        
        sheet_itc_aud.write(1+subjN, 0, subj) 
        sheet_itc_aud.write(1+subjN, 1, str(peaksAuditoryLeft[0])) 
        sheet_itc_aud.write(1+subjN, 2, str(peaksAuditoryLeft[1]))
        sheet_itc_aud.write(1+subjN, 3, str(peaksAuditoryLeft[2]))
        sheet_itc_aud.write(1+subjN, 4, str(peaksAuditoryLeft[3]))
        sheet_itc_aud.write(1+subjN, 5, str(peaksAuditoryLeft[4]))
        sheet_itc_aud.write(1+subjN, 7, str(peaksAuditoryRight[0])) 
        sheet_itc_aud.write(1+subjN, 8, str(peaksAuditoryRight[1]))
        sheet_itc_aud.write(1+subjN, 9, str(peaksAuditoryRight[2]))
        sheet_itc_aud.write(1+subjN, 10, str(peaksAuditoryRight[3]))
        sheet_itc_aud.write(1+subjN, 11, str(peaksAuditoryRight[4]))
        sheet_itc_aud.write(1+subjN, 13, str(peaksAuditoryBoth[0])) 
        sheet_itc_aud.write(1+subjN, 14, str(peaksAuditoryBoth[1]))
        sheet_itc_aud.write(1+subjN, 15, str(peaksAuditoryBoth[2]))
        sheet_itc_aud.write(1+subjN, 16, str(peaksAuditoryBoth[3]))
        sheet_itc_aud.write(1+subjN, 17, str(peaksAuditoryBoth[4])) 
        
        sheet_amp_all.write(1+subjN, 0, subj) 
        sheet_amp_all.write(1+subjN, 1, str(ampAllLeft[0])) 
        sheet_amp_all.write(1+subjN, 2, str(ampAllLeft[1]))
        sheet_amp_all.write(1+subjN, 3, str(ampAllLeft[2]))
        sheet_amp_all.write(1+subjN, 4, str(ampAllLeft[3]))
        sheet_amp_all.write(1+subjN, 5, str(ampAllLeft[4]))
        sheet_amp_all.write(1+subjN, 7, str(ampAllRight[0])) 
        sheet_amp_all.write(1+subjN, 8, str(ampAllRight[1]))
        sheet_amp_all.write(1+subjN, 9, str(ampAllRight[2]))
        sheet_amp_all.write(1+subjN, 10, str(ampAllRight[3]))
        sheet_amp_all.write(1+subjN, 11, str(ampAllRight[4]))
        sheet_amp_all.write(1+subjN, 13, str(ampAllBoth[0])) 
        sheet_amp_all.write(1+subjN, 14, str(ampAllBoth[1]))
        sheet_amp_all.write(1+subjN, 15, str(ampAllBoth[2]))
        sheet_amp_all.write(1+subjN, 16, str(ampAllBoth[3]))
        sheet_amp_all.write(1+subjN, 17, str(ampAllBoth[4])) 
        
        sheet_amp_aud.write(1+subjN, 0, subj) 
        sheet_amp_aud.write(1+subjN, 1, str(ampAudLeft[0])) 
        sheet_amp_aud.write(1+subjN, 2, str(ampAudLeft[1]))
        sheet_amp_aud.write(1+subjN, 3, str(ampAudLeft[2]))
        sheet_amp_aud.write(1+subjN, 4, str(ampAudLeft[3]))
        sheet_amp_aud.write(1+subjN, 5, str(ampAudLeft[4]))
        sheet_amp_aud.write(1+subjN, 7, str(ampAudRight[0])) 
        sheet_amp_aud.write(1+subjN, 8, str(ampAudRight[1]))
        sheet_amp_aud.write(1+subjN, 9, str(ampAudRight[2]))
        sheet_amp_aud.write(1+subjN, 10, str(ampAudRight[3]))
        sheet_amp_aud.write(1+subjN, 11, str(ampAudRight[4]))
        sheet_amp_aud.write(1+subjN, 13, str(ampAudBoth[0])) 
        sheet_amp_aud.write(1+subjN, 14, str(ampAudBoth[1]))
        sheet_amp_aud.write(1+subjN, 15, str(ampAudBoth[2]))
        sheet_amp_aud.write(1+subjN, 16, str(ampAudBoth[3]))
        sheet_amp_aud.write(1+subjN, 17, str(ampAudBoth[4])) 
        
        sheet_num_trials.write(1+subjN, 0, subj)
        sheet_num_trials.write(1+subjN, 1, str(projN))
        sheet_num_trials.write(1+subjN, 2, str(len(epochs_avg['1'])))
        sheet_num_trials.write(1+subjN, 3, str(len(epochs_avg['2'])))
        sheet_num_trials.write(1+subjN, 4, str(len(epochs_avg['3'])))
        sheet_num_trials.write(1+subjN, 5, str(len(epochs_avg['4'])))
        sheet_num_trials.write(1+subjN, 6, str(len(epochs_avg['5'])))
        sheet_num_trials.write(1+subjN, 7, str(len(epochs_avg['6'])))
        sheet_num_trials.write(1+subjN, 8, str(len(epochs_avg['7'])))
        sheet_num_trials.write(1+subjN, 9, str(len(epochs_avg['8'])))
        
        wb.save(froot+'/EEGdata.xls')
    
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