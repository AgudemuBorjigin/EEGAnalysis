#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 22:27:01 2019

@author: agudemu
"""
from anlffr.helper import biosemi2mne as bs
from anlffr.preproc import find_blinks #
from mne.preprocessing.ssp import compute_proj_epochs #
import mne
import numpy as np # support for large, multi-dimensional arrays and metrices
import pylab as pl
import os # it assigns its path attribute to an os-specific path module
import fnmatch # unix filename pattern matching
#import xlwt

############################pre-processing of raw data#######################################################################################
def preprocessing_raw(fpath, subj):
    # load data and read event channels
    rawlist = []
    evelist = []
    # extracting bdf filenames into bdfs
    bdfs = fnmatch.filter(os.listdir(fpath), subj + '_ITD*.bdf') 
    
    if len(bdfs) >= 1:
        for k, bdf in enumerate(bdfs):
            rawtemp, evestemp = bs.importbdf(fpath + bdf)
            rawlist += [rawtemp,]
            evelist += [evestemp,]
    else:
        RuntimeError("No bdf files found!")
    raw, eves = mne.concatenate_raws(rawlist, events_list = evelist)
    # Things to try if data are too noisy
    # raw.info['bads'] += ['A20'] # if A7, A6, A24 have been rejected a lot, they can be added manually
    # raw.set_eeg_reference(ref_channels = ['Average']) # referencing to the average of all channels, do this if channels are too noisy  
    
    # the returned eves has three columns: sample number, value of onset, value of offset.     
    eves2 = eves.copy()
    eves2[:, 1] = np.mod(eves2[:, 1], 256) 
    eves2[:, 2] = np.mod(eves2[:, 2], 256)   
    
    # raw.plot(events=eves2)
    raw.filter(l_freq = 1, h_freq=50) # if channels are noisy, adjust the filter parameters, high-pass frequency is better set at 1 compared to 0.5 Hz, to get rid of low-frequency drift
    
    # SSP for blinks
    blinks = find_blinks(raw, ch_name = ['A1',], l_trans_bandwidth=0.5,
                         l_freq=1.0) # A1 is the closest electrode to eyebrow, raw.plot(events=blinks) shows the lines at eye blinks, 998 is default event_id
    epochs_blinks = mne.Epochs(raw, blinks, 998, tmin = -0.25, tmax = 0.25, 
                               proj = False, baseline = (-0.25, 0), 
                               reject=dict(eeg=500e-6)) 
    n_eeg = 4
    blink_projs = compute_proj_epochs(epochs_blinks, n_grad=0,
                                      n_mag=0, n_eeg=n_eeg,
                                      verbose='DEBUG')    
    if subj in ['S025', 'S031', 'S046', 'S117', 'S123', 'S127', 'S128', 'S132', 'S133', 'S143', 'S149', 'S051', 'S183', 'S185', 'S187', 'S189', 'S193', 'S195', 'S196', 'S043', 'S072', 'S078', 'S191', 'S129', 'S141', 'S190', 'S194']:
        #raw.add_proj(blink_projs) # adding all projections
        #raw.del_proj()
        #raw.plot_projs_topomap()
        raw.add_proj([blink_projs[0], blink_projs[2]])   
    elif subj in ['S135', 'S192', 'S197', 'S199', 'S216', 'S218']:
        raw.add_proj([blink_projs[0], blink_projs[1]])   
    elif subj in ['S084', 'S075']:
        raw.add_proj([blink_projs[0]])
    return raw, eves2

###########################extracting the average response###################################################################################
def evoked(raw, eves2, triggerID):
    epochs = mne.Epochs(raw, eves2, triggerID, tmin = -0.5, proj = True, tmax = 2.5, 
                    baseline = (-0.5, 0), reject = dict(eeg=150e-6)) # change the channels as needed
    evoked_raw = epochs.average()
    # see which channels are bad by using evoked.plot(picks=[30, 31]) and evoked.plot_topomap(times=[1.2]), evoked.pick_chanels()
    return evoked_raw

###########################mag-lat calculation###############################################################################################
def indexExtraction(t, timeWindow):
    index = np.where(t>timeWindow[0])
    index1 = index[0]
    index1 = index1[0]
    index = np.where(t<timeWindow[1]) 
    index2 = index[0]
    index2 = index2[-1]
    return index1, index2

def LatMag(evoked, t, timeWindowOnset, timeWindowITD, cond, subj):
    index1, index2 = indexExtraction(t, timeWindowITD)
    n1AmpITD = min(evoked[index1:index2])
    p2AmpITD = max(evoked[index1:index2])
    stdITD = np.std(evoked[index1:index2])
    meanITD = np.mean(evoked[index1:index2])
    indexMin = np.where(evoked == n1AmpITD)
    indexMax = np.where(evoked == p2AmpITD)
    # latency from ITD shift to ERP
    index = np.where(t>0.9804)
    indexRef = index[0]
    indexRef = indexRef[0]
    n1Lat = t[indexMin] - t[indexRef]
    n1Lat = round(n1Lat, 4)
    p2Lat = t[indexMax] - t[indexRef]
    p2Lat = round(p2Lat, 4)
    # noise rejection
    if n1AmpITD > meanITD - stdITD:
        n1AmpITD = np.nan
        n1Lat = np.nan
    if p2AmpITD < meanITD + stdITD:
        p2AmpITD = np.nan
        p2Lat = np.nan
    # onset ERP: P2, N1 mag
    index1, index2 = indexExtraction(t, timeWindowOnset)
    n1AmpOnset = min(evoked[index1:index2])
    p2AmpOnset = max(evoked[index1:index2])
    meanOnset = np.mean(evoked[index1:index2])
    stdOnset = np.std(evoked[index1:index2])
    if n1AmpOnset > (meanOnset - stdOnset):
        n1AmpOnset = np.nan
    if p2AmpOnset < (meanOnset + stdOnset):
        p2AmpOnset = np.nan
    plotERP(evoked, stdOnset, meanOnset, stdITD, meanITD, cond, subj, timeWindowOnset, timeWindowITD)
    return n1Lat, p2Lat, n1AmpITD, p2AmpITD, n1AmpOnset, p2AmpOnset

def plotERP(evokedRes, stdOnset, meanOnset, stdITD, meanITD, title, subj, timeWindowOnset, timeWindowITD):
    fig = pl.figure(figsize = (20, 5))
    ax = fig.add_subplot(111)
    ax.plot(t, evokedRes)
    pl.title(subj+ '-' + title)
    # annotation of peak points
    annot_max_min(t, evokedRes, timeWindowOnset, 'max', 'Onset')
    annot_max_min(t, evokedRes, timeWindowOnset, 'min', 'Onset')
    annot_max_min(t, evokedRes, timeWindowITD, 'max', 'ITD')
    annot_max_min(t, evokedRes, timeWindowITD, 'min', 'ITD')
    # drawing lines for time window
    pl.axvline(x = timeWindowOnset[0], color = 'r')
    pl.axvline(x = timeWindowOnset[1], color = 'r')
    pl.axvline(x = timeWindowITD[0], color = 'k')
    pl.axvline(x = timeWindowITD[1], color = 'k')
    # drawing 1-std lines above and below the mean
    pl.axhline(y = meanOnset + stdOnset, color = 'r', linestyle = 'dashed')
    pl.axhline(y = meanOnset, color = 'r')
    pl.text(0, meanOnset, str(meanOnset))
    pl.axhline(y = meanOnset - stdOnset, color = 'r', linestyle = 'dashed')
    # drawing 1-std lines above and below the mean
    pl.axhline(y = meanITD + stdITD, color = 'k', linestyle = 'dashed')
    pl.axhline(y = meanITD, color = 'k')
    pl.text(0, meanITD, str(meanITD))
    pl.axhline(y = meanITD - stdITD, color = 'k', linestyle = 'dashed')
    # saving the image
    pl.savefig(froot + '/ERPfigures/' + subj + '/' + subj + '-' + title + '.png')
    
def annot_max_min(t, evoked, timeWindow, type_max_min, where):
    index1, index2 = indexExtraction(t, timeWindow)
    if type_max_min is 'max':
        e = max(evoked[index1:index2])
        index = np.where(evoked == e)
        tPoint = t[index]
    elif type_max_min is 'min':
        e = min(evoked[index1:index2])
        index = np.where(evoked == e)
        tPoint = t[index]
    if where is 'ITD':
        mkrClr = 'r'
    else:
        mkrClr = 'k' 
    pl.plot(tPoint, e, mkrClr + '+', markersize = 12, linewidth = 8)

# ITD ERP: P2-N1 mag, normalized by onset ERP P2-N1 mag
def normRatio(n1ampITD, p2ampITD, n1ampOnset, p2ampOnset):
    n1p2ampITD = p2ampITD - n1ampITD
    n1p2ampOnset = p2ampOnset - n1ampOnset
    n1p2ampITDNor = n1p2ampITD/n1p2ampOnset
    return round(n1p2ampITDNor, 3)
    
#######################################################################################################################################################
stimulus = 'ITD'
OS = 'Ubuntu'

if OS == 'Ubuntu':
    froot = '/media/agudemu/Storage/Data/EEG/ITD'
else:
    froot = '/Users/baoagudemu1/Desktop/Lab/Experiment/DataAnalysis/Data' 

subjOrder = 0
#sublist = ['S025', 'S031', 'S043', 'S046', 'S051', 'S072', 'S075', 'S078', 'S084', 'S117', 'S123', 'S127', 
#         'S128', 'S132', 'S133', 'S135', 'S143', 'S149', 'S183', 'S185', 'S187', 'S189', 'S191', 'S192', 
#         'S193', 'S194', 'S195', 'S196', 'S197', 'S199', 'S216', 'S218']
sublist = ['S218']
for subj in sublist:
    subjOrder = subjOrder + 1
    os.mkdir(froot + '/ERPfigures/' + subj)
    fpath = froot + '/' + subj + '/'
    
    raw, eves2 = preprocessing_raw(fpath, subj)    
        
    evokedAvg = evoked(raw, eves2, [1, 5, 2, 6, 3, 7, 4, 8])
    evoked20us = evoked(raw, eves2, [1, 5])
    evoked60us = evoked(raw, eves2, [2, 6])
    evoked180us = evoked(raw, eves2, [3, 7])
    evoked540us = evoked(raw, eves2, [4, 8])
    
    t = evoked20us.times
    # time window for onse and ITD-evoked response
    timeWindowOnset = [0.08, 0.25]  
    timeWindowITD = [1.1, 1.22]
#    evokedAvg.plot(picks = [30, 31]) 
    channNum = 31
    
    
    evokedAvgLP = evokedAvg.data[channNum]
    evoked20usLP = evoked20us.data[channNum]
    evoked60usLP = evoked60us.data[channNum]
    evoked180usLP = evoked180us.data[channNum]
    evoked540usLP = evoked540us.data[channNum]
    
    n1Lat20, p2Lat20, n1ampITD20, p2ampITD20, n1ampOnset20, p2ampOnset20 = LatMag(evoked20usLP, t, [0.09, 0.23], [1.13, 1.28], '20us', subj)
    n1p2ITDnor20 = normRatio(n1ampITD20, p2ampITD20, n1ampOnset20, p2ampOnset20) 
    
    n1Lat60, p2Lat60, n1ampITD60, p2ampITD60, n1ampOnset60, p2ampOnset60 = LatMag(evoked60usLP, t, [0.10, 0.19], [1.08, 1.25], '60us', subj)
    n1p2ITDnor60 = normRatio(n1ampITD60, p2ampITD60, n1ampOnset60, p2ampOnset60)
    
    n1Lat180, p2Lat180, n1ampITD180, p2ampITD180, n1ampOnset180, p2ampOnset180 = LatMag(evoked180usLP, t, [0.09, 0.18], [1.08, 1.22], '180us', subj)
    n1p2ITDnor180 = normRatio(n1ampITD180, p2ampITD180, n1ampOnset180, p2ampOnset180)   
    
    n1Lat540, p2Lat540, n1ampITD540, p2ampITD540, n1ampOnset540, p2ampOnset540 = LatMag(evoked540usLP, t, [0.1, 0.21], [1.07, 1.23], '540us', subj)
    n1p2ITDnor540 = normRatio(n1ampITD540, p2ampITD540, n1ampOnset540, p2ampOnset540)
    
#    n1LatAvg, p2LatAvg, n1ampITDAvg, p2ampITDAvg, n1ampOnsetAvg, p2ampOnsetAvg = LatMag(evokedAvgLP, t, timeWindowOnset, timeWindowITD, 'Avg', subj)
#    n1p2ITDnor = normRatio(n1ampITDAvg, p2ampITDAvg, n1ampOnsetAvg, p2ampOnsetAvg)
#    if subjOrder == 1:
#        evoked180us_avg = np.zeros(len(evoked180usLP))
#    evoked180us_avg = evoked180us_avg + evoked180usLP
#
#evoked180us_avg = evoked180us_avg / subjOrder 
#
#fontSize = 35
#fig = pl.figure(figsize = (20, 15))
#pl.plot(t, evoked180us_avg*1e6, 'b')
#annot_max_min(t, evoked180us_avg*1e6, timeWindowITD, 'max', 'ITD')
#annot_max_min(t, evoked180us_avg*1e6, timeWindowITD, 'min', 'ITD')
#pl.axvline(x = 0.98, color = 'r')   
#pl.xlabel('Time [s]', fontsize=fontSize)
#pl.ylabel('Evoked potential [uV]', fontsize=fontSize)
#pl.rc('xtick',labelsize=fontSize)
#pl.rc('ytick',labelsize=fontSize)  
    
