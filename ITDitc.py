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
#from scipy.signal import butter, lfilter
import xlwt


def noiseFloorEstimate(t, evoked, timeWindow):
    index = np.where(t > timeWindow[0])
    index1 = index[0]
    index1 = index1[0]
    index = np.where(t < timeWindow[1]) 
    index2 = index[0]
    index2 = index2[-1]
    
    noiseFloor = evoked[0:index1].mean(axis=0)
    return noiseFloor, index1, index2

def indexExtraction(t, timeWindow):
    index = np.where(t>timeWindow[0])
    index1 = index[0]
    index1 = index1[0]
    index = np.where(t<timeWindow[1]) 
    index2 = index[0]
    index2 = index2[-1]
    return index1, index2

def itc_normalization(stimulus, itc, freqs, t, cond, timeWindowOnset, timeWindowITD):
    if stimulus == 'ITD':
        freqSub = np.where(freqs<20)
    elif stimulus == 'Atten':
        freqSub = np.where(freqs<50) 
        
    itc_ave = itc[freqSub[0], :].mean(axis=0)
    noiseFloor, index1, index2 = noiseFloorEstimate(t, itc_ave, timeWindowOnset)
    itc_ave = itc_ave - noiseFloor
    
    firstPeakAmp = np.absolute(np.max(itc_ave[index1:index2])) 
    itc_norm = itc_ave/firstPeakAmp
    
    index1, index2 = indexExtraction(t, timeWindowITD)
    stdITD = np.std(itc_norm[index1:index2:])
    meanITD = np.mean(itc_norm[index1:index2])
    plot_avg(itc_norm, t, cond, timeWindowOnset, timeWindowITD, meanITD, stdITD)
    
    return itc_norm, meanITD, stdITD

def peak(itc_norm, t, timeWindow):
    index = np.where(t > timeWindow[0])
    index1 = index[0]
    index1 = index1[0]
    index = np.where(t < timeWindow[1]) 
    index2 = index[0]
    index2 = index2[-1]
    
    # Latency from ITD shift to ERP
    index = np.where(t>0.9804)
    indexRef = index[0]
    indexRef = indexRef[0]
    maxITC = max(itc_norm[index1:index2])
    indexMax = np.where(itc_norm == maxITC)
    Lat = t[indexMax] - t[indexRef]
    Lat = round(Lat, 4)
    
    return maxITC, Lat

def itc(t, epochs, cond, channels):
    # channels in the format of [4, 26, 25, 30, 31], auditory channels
    # itc_data_mean_all = itc_data.mean(axis = 0) # CHANGE AS NEEDED: average across all channels 
    
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
    itc_data_mean = itc_data[channels, :, :].mean(axis = 0) 
    
    np.savez(froot+'/'+'itcCz' + '/'+subj+'_'+'itc'+str(cond), itc = itc_data, t = t, freqs = freqs, itc_avg = itc_data_mean); 
    # npzFile = np.load(fpath+'/'+'itc_power.npz'), npzFiles.files, npzFile['itc']
    # if bad channels were added manually, select good channels
    power_copy = power.copy()
    power_data = power_copy.data
    np.savez(froot+'/'+'powerCz' + '/'+subj+'_'+'power'+str(cond), power = power_data, t = t, freqs = freqs);
    
    plot_spectrogram(itc_data_mean, t, freqs, cond)
    pl.savefig(froot + '/ITCfigures/' + subj + '/' + subj + '_spectrogram_' + cond + '.png')
    
    return itc_data_mean

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
   
def plot_avg(itc_norm, t, cond, timeWindowOnset, timeWindowITD, meanITD, stdITD):
    fig = pl.figure(figsize = (20, 5))
    ax = fig.add_subplot(111)
    ax.plot(t, itc_norm) 
    pl.axvline(x = timeWindowITD[0], color = 'r')
    pl.axvline(x = timeWindowITD[1], color = 'r')
    pl.axvline(x = timeWindowOnset[0], color = 'r')
    pl.axvline(x = timeWindowOnset[1], color = 'r')
    pl.xlabel('Time (s)', fontsize=14)
    pl.ylabel('Normalized response', fontsize=14)
    titleStr = subj + '_' + str(cond)
    pl.title(titleStr, fontsize=14)
    ax = pl.gca()
    ax.tick_params(labelsize=14)
    
    pl.axhline(y = meanITD + stdITD, color = 'k', linestyle = 'dashed')
    pl.axhline(y = meanITD, color = 'k')
    pl.text(0, meanITD, str(meanITD))
    pl.axhline(y = meanITD - stdITD, color = 'k', linestyle = 'dashed')
    
    peakITD, _ = peak(itc_norm, t, timeWindowITD)
    index = np.where(itc_norm == peakITD)
    tPoint = t[index]
    ax.plot(tPoint, peakITD, 'r+', markersize = 12, linewidth = 8)
    pl.savefig(froot + '/ITCfigures/' + subj + '/' + titleStr + '.png')
            
def auditoryAvg(evoked):
    evokedAvg = np.zeros(shape = (1, len(evoked.data[0])))
    audChanns = [4, 26, 25, 30, 31]
    for i in audChanns:
        evokedAvg = evokedAvg + evoked.data[i]
    evokedAvg = evokedAvg / len(audChanns)
    return evokedAvg[0]
   
def evoked(raw, eves2, triggerID):
    epochs = mne.Epochs(raw, eves2, triggerID, tmin = -0.5, proj = True, tmax = 2.5, 
                    baseline = (-0.5, 0), reject = dict(eeg=150e-6)) # change the channels as needed
    # always start with looking at evoked (averaged) response, 
    # and see which channels are bad by using evoked.plot(picks=[30, 31]) and evoked.plot_topomap(times=[1.2]), evoked.pick_channels()
    # add those channels to the list manually 
    # by raw.info['bads'] += ['A7', 'A6', 'A24'] if necessary
    # evoked_raw = epochs.average()
    # noiseFloor, _, _, = noiseFloorEstimate(t, evoked.data[31]) # noise floor is very small, since DC has been filtered out
    # evoked_chann32 = evoked.data[31] - noiseFloor
    # evokedAll = evoked_raw.data.mean(axis = 0)
    epochs.drop_bad()
    return epochs


def processing(stimulus, subj, fpath):
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
        raw.filter(l_freq = 1, h_freq=50) # if channels are noisy, adjust the filter parameters
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
    # evoked_blinks = epochs_blinks.average()
    # evoked_blinks_data = evoked_blinks.data[np.arange(32),:]
    
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
    if subj in ['S025', 'S031', 'S046', 'S117', 'S123', 'S127', 'S128', 'S132', 'S133', 'S143', 'S149', 'S051', 'S183', 'S185', 'S187', 'S189', 'S193', 'S195', 'S196', 'S043', 'S072', 'S075', 'S078', 'S191', 'S129', 'S141', 'S190', 'S021', 'S024', 'S028', 'S083', 'S119', 'S121', 'S122', 'S139', 'S140', 'S142', 'S144', 'S145']:
        #raw.add_proj(blink_projs) # adding all projections
        raw.add_proj([blink_projs[0], blink_projs[2]]) # raw.del_proj()   
        projN = [1, 3]
    elif subj in ['S135', 'S192', 'S194', 'S197', 'S199', 'S216', 'S218']:
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

def dataITC(t, epochs1, epochs2, epochs3, cond1, cond2, cond3, channels):
    itcMean = itc(t, epochs1, cond1, channels)
    itcMeanleft = itc(t, epochs2, cond2, channels)
    itcMeanright = itc(t, epochs3, cond3, channels)
    return itcMean, itcMeanleft, itcMeanright

def peakITC(itcMeanType, freqs, t, conds, timeWindowOnset, timeWindowITD):
    peaks = [0]*len(itcMeanType)
    lats = [0]*len(itcMeanType)
    for i in range(len(itcMeanType)):
        itc_norm, meanITD, stdITD = itc_normalization(stimulus, itcMeanType[i], freqs, t, conds[i], timeWindowOnset, timeWindowITD)
        peaks[i], lats[i] = peak(itc_norm, t, timeWindowITD)
        if peaks[i] < meanITD + stdITD:
            peaks[i] = np.nan
            lats[i] = np.nan
        else:
            peaks[i] = round(peaks[i], 4)
            lats[i] = round(lats[i], 4)
    return peaks, lats

def dataLoading(subj, cond):
    both = np.load(froot + '/' + 'itcCz/' + subj + '_itc' + cond + '.npz')
    left = np.load(froot + '/' + 'itcCz/' +subj + '_itc' + cond + '_left.npz')
    right = np.load(froot + '/' + 'itcCz/' +subj + '_itc' + cond + '_right.npz')
    
    return both['itc_avg'], left['itc_avg'], right['itc_avg'], both['t']
##############################################################################################################################################################

stimulus = 'ITD'
OS = 'Ubuntu'

#subjlist = ['S021', 'S133', 'S135', 'S143', 'S149', 'S183', 'S185', 'S187', 'S189', 'S191', 'S192', 'S193', 'S194', 'S195', 'S196', 
#            'S197', 'S199', 'S024', 'S028', 'S083', 'S119', 'S139', 'S140', 'S142', 'S144', 'S145', 'S216', 'S218'];
#subjlist = ['S025', 'S031', 'S043', 'S046', 'S051', 'S072', 'S075', 'S078', 'S084', 'S117', 'S123', 'S127', 'S128', 'S132']
subjlist = ['S218']


if stimulus == 'ITD':
    if OS == 'Ubuntu':
        froot = '/media/agudemu/Storage/Data/EEG/ITD'
    else:
        froot = '/Users/baoagudemu1/Desktop/Lab/Experiment/DataAnalysis/Data'    
    wb = xlwt.Workbook()
    sheet_itc_all = wb.add_sheet('itc_Cz')
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
    sheet_itc_all.write(0, 19, 'Lat-left', style_bold) 
    sheet_itc_all.write(0, 25, 'Lat-right', style_bold) 
    sheet_itc_all.write(0, 31, 'Lat-both', style_bold) 
    sheet_itc_all.write(1, 1, '20 us', style_bold) 
    sheet_itc_all.write(1, 2, '60 us', style_bold)
    sheet_itc_all.write(1, 3, '180 us', style_bold)
    sheet_itc_all.write(1, 4, '540 us', style_bold)
    sheet_itc_all.write(1, 5, 'Avg', style_bold)
    sheet_itc_all.write(1, 6, 'AvgExd20', style_bold)
    
    sheet_itc_all.write(1, 7, '20 us', style_bold) 
    sheet_itc_all.write(1, 8, '60 us', style_bold)
    sheet_itc_all.write(1, 9, '180 us', style_bold)
    sheet_itc_all.write(1, 10, '540 us', style_bold)
    sheet_itc_all.write(1, 11, 'Avg', style_bold)
    sheet_itc_all.write(1, 12, 'AvgExd20', style_bold)
    
    sheet_itc_all.write(1, 13, '20 us', style_bold) 
    sheet_itc_all.write(1, 14, '60 us', style_bold)
    sheet_itc_all.write(1, 15, '180 us', style_bold)
    sheet_itc_all.write(1, 16, '540 us', style_bold)
    sheet_itc_all.write(1, 17, 'Avg', style_bold)
    sheet_itc_all.write(1, 18, 'AvgExd20', style_bold)
    
    sheet_itc_all.write(1, 19, '20 us', style_bold) 
    sheet_itc_all.write(1, 20, '60 us', style_bold)
    sheet_itc_all.write(1, 21, '180 us', style_bold)
    sheet_itc_all.write(1, 22, '540 us', style_bold)
    sheet_itc_all.write(1, 23, 'Avg', style_bold)
    sheet_itc_all.write(1, 24, 'AvgExd20', style_bold)
    
    sheet_itc_all.write(1, 25, '20 us', style_bold) 
    sheet_itc_all.write(1, 26, '60 us', style_bold)
    sheet_itc_all.write(1, 27, '180 us', style_bold)
    sheet_itc_all.write(1, 28, '540 us', style_bold)
    sheet_itc_all.write(1, 29, 'Avg', style_bold)
    sheet_itc_all.write(1, 30, 'AvgExd20', style_bold)
    
    sheet_itc_all.write(1, 31, '20 us', style_bold) 
    sheet_itc_all.write(1, 32, '60 us', style_bold)
    sheet_itc_all.write(1, 33, '180 us', style_bold)
    sheet_itc_all.write(1, 34, '540 us', style_bold)
    sheet_itc_all.write(1, 35, 'Avg', style_bold)
    sheet_itc_all.write(1, 36, 'AvgExd20', style_bold)
    
    wb.save(froot+'/EEG_ITC_Cz.xls')
elif stimulus == 'Atten':
    if OS == 'Ubuntu':
        froot = '/media/agudemu/Storage/Data/EEG/Atten'
    else:
        froot = '/Users/baoagudemu1/Desktop/Lab/EEG-Python/Atten'

subjN = 0
for subj in subjlist:
    subjN = subjN + 1
    os.mkdir(froot + '/ITCfigures/' + subj)
    fpath = froot + '/' + subj + '/'
    print 'Running subject', subj
    
    [raw, eves2, projN] = processing(stimulus, subj, fpath)
    
    if stimulus == 'ITD':
        epochs_avg = evoked(raw, eves2, [1, 2, 3, 4, 5, 6, 7, 8])
#        epochs_avg_left = evoked(raw, eves2, [1, 2, 3, 4])
#        epochs_avg_right = evoked(raw, eves2, [5, 6, 7, 8])
#        epochs_avgExd20 = evoked(raw, eves2, [2, 3, 4, 6, 7, 8])
#        epochs_avgExd20_left = evoked(raw, eves2, [2, 3, 4])
#        epochs_avgExd20_right = evoked(raw, eves2, [6, 7, 8])
#        epochs1 = evoked(raw, eves2, [1, 5])
#        epochs1_left = evoked(raw, eves2, [1])
#        epochs1_right = evoked(raw, eves2, [5])
#        epochs2 = evoked(raw, eves2, [2, 6])
#        epochs2_left = evoked(raw, eves2, [2])
#        epochs2_right = evoked(raw, eves2, [6])
#        epochs3 = evoked(raw, eves2, [3, 7])
#        epochs3_left = evoked(raw, eves2, [3])
#        epochs3_right = evoked(raw, eves2, [7])
#        epochs4 = evoked(raw, eves2, [4, 8])
#        epochs4_left = evoked(raw, eves2, [4])
#        epochs4_right = evoked(raw, eves2, [8])
#        
#        t = epochs_avg.times
#        freqs = np.arange(1., 50., 1.) # chaneg as needed
        
#        # ITC processing
#        itcMeanAll, itcMeanAll_left, itcMeanAll_right = dataITC(t, epochs_avg, epochs_avg_left, epochs_avg_right, 'avg', 'avg_left', 'avg_right', [31])
#        
#        itcMeanAllExd20, itcMeanAllExd20_left, itcMeanAllExd20_right = dataITC(t, epochs_avgExd20, epochs_avgExd20_left, epochs_avgExd20_right, 'avgExd20', 'avgExd20_left', 'avgExd20_right', [31])
#        
#        itcMean20, itcMean20_left, itcMean20_right = dataITC(t, epochs1, epochs1_left, epochs1_right, '20us', '20us_left', '20us_right', [31])
#        
#        itcMean60, itcMean60_left, itcMean60_right = dataITC(t, epochs2, epochs2_left, epochs2_right, '60us', '60us_left', '60us_right', [31])
#        
#        itcMean180, itcMean180_left, itcMean180_right = dataITC(t, epochs3, epochs3_left, epochs3_right, '180us', '180us_left', '180us_right', [31])
#        
#        itcMean540, itcMean540_left, itcMean540_right = dataITC(t, epochs4, epochs4_left, epochs4_right, '540us', '540us_left', '540us_right', [31])
        
        # Extraction of processed ITC data
        freqs = np.arange(1., 50., 1.) # chaneg as needed
        [itcMean20, itcMean20_left, itcMean20_right, t] = dataLoading(subj, '20us')
        [itcMean60, itcMean60_left, itcMean60_right, t] = dataLoading(subj, '60us')
        [itcMean180, itcMean180_left, itcMean180_right, t] = dataLoading(subj, '180us')
        [itcMean540, itcMean540_left, itcMean540_right, t] = dataLoading(subj, '540us')
        [itcMeanAll, itcMeanAll_left, itcMeanAll_right, t] = dataLoading(subj, 'avg')
        [itcMeanAllExd20, itcMeanAllExd20_left, itcMeanAllExd20_right, t] = dataLoading(subj, 'avgExd20')
        
        itcMeanType = [itcMean20, itcMean60, itcMean180, itcMean540, itcMeanAll, itcMeanAllExd20]
        conds = ['20usBoth', '60usBoth', '180usBoth', '540usBoth', 'AvgBoth', 'AvgExd20Both']
        peaksBoth, latBoth = peakITC(itcMeanType, freqs, t, conds, [0, 0.3], [1.1, 1.2])
       
        itcMeanType = [itcMean20_left, itcMean60_left, itcMean180_left, itcMean540_left, itcMeanAll_left, itcMeanAllExd20_left]
        conds = ['20usLeft', '60usLeft', '180usLeft', '540usLeft', 'AvgLeft', 'AvgExd20Left']
        peaksLeft, latLeft = peakITC(itcMeanType, freqs, t, conds, [0, 0.3], [1.1, 1.2])
            
        itcMeanType = [itcMean20_right, itcMean60_right, itcMean180_right, itcMean540_right, itcMeanAll_right, itcMeanAllExd20_right]
        conds = ['20usRight', '60usRight', '180usRight', '540usRight', 'AvgRight', 'AvgExd20Right']
        peaksRight, latRight = peakITC(itcMeanType, freqs, t, conds, [0, 0.3], [1.1, 1.2])

        
        # storing data to spreadsheet
        sheet_itc_all.write(1+subjN, 0, subj) 
        sheet_itc_all.write(1+subjN, 1, str(peaksLeft[0])) 
        sheet_itc_all.write(1+subjN, 2, str(peaksLeft[1]))
        sheet_itc_all.write(1+subjN, 3, str(peaksLeft[2]))
        sheet_itc_all.write(1+subjN, 4, str(peaksLeft[3]))
        sheet_itc_all.write(1+subjN, 5, str(peaksLeft[4]))
        sheet_itc_all.write(1+subjN, 6, str(peaksLeft[5]))
        sheet_itc_all.write(1+subjN, 7, str(peaksRight[0])) 
        sheet_itc_all.write(1+subjN, 8, str(peaksRight[1]))
        sheet_itc_all.write(1+subjN, 9, str(peaksRight[2]))
        sheet_itc_all.write(1+subjN, 10, str(peaksRight[3]))
        sheet_itc_all.write(1+subjN, 11, str(peaksRight[4]))
        sheet_itc_all.write(1+subjN, 12, str(peaksRight[5]))
        sheet_itc_all.write(1+subjN, 13, str(peaksBoth[0])) 
        sheet_itc_all.write(1+subjN, 14, str(peaksBoth[1]))
        sheet_itc_all.write(1+subjN, 15, str(peaksBoth[2]))
        sheet_itc_all.write(1+subjN, 16, str(peaksBoth[3]))
        sheet_itc_all.write(1+subjN, 17, str(peaksBoth[4])) 
        sheet_itc_all.write(1+subjN, 18, str(peaksBoth[5])) 
        sheet_itc_all.write(1+subjN, 19, str(latLeft[0])) 
        sheet_itc_all.write(1+subjN, 20, str(latLeft[1]))
        sheet_itc_all.write(1+subjN, 21, str(latLeft[2]))
        sheet_itc_all.write(1+subjN, 22, str(latLeft[3]))
        sheet_itc_all.write(1+subjN, 23, str(latLeft[4]))
        sheet_itc_all.write(1+subjN, 24, str(latLeft[5])) 
        sheet_itc_all.write(1+subjN, 25, str(latRight[0]))
        sheet_itc_all.write(1+subjN, 26, str(latRight[1]))
        sheet_itc_all.write(1+subjN, 27, str(latRight[2]))
        sheet_itc_all.write(1+subjN, 28, str(latRight[3]))
        sheet_itc_all.write(1+subjN, 29, str(latRight[4]))
        sheet_itc_all.write(1+subjN, 30, str(latRight[5]))
        sheet_itc_all.write(1+subjN, 31, str(latBoth[0])) 
        sheet_itc_all.write(1+subjN, 32, str(latBoth[1]))
        sheet_itc_all.write(1+subjN, 33, str(latBoth[2]))
        sheet_itc_all.write(1+subjN, 34, str(latBoth[3]))
        sheet_itc_all.write(1+subjN, 35, str(latBoth[4])) 
        sheet_itc_all.write(1+subjN, 36, str(latBoth[5])) 
        
        
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
        
        wb.save(froot+'/EEG_ITC_Cz.xls')
    
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