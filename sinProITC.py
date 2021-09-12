#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 12:20:16 2019

@author: agudemu
"""

import numpy as np # support for large, multi-dimensional arrays and metrices
import pylab as pl
from scipy.signal import butter, lfilter
import xlwt

def itcAvgExtraction(conds):
    itc_20us = np.load(froot + subj + '_' + conds[0] + '.npz')
    itcMean20 = itc_20us['itc_avg']

    itc_60us = np.load(froot + subj + '_' + conds[1] + '.npz')
    itcMean60 = itc_60us['itc_avg']

    itc_180us = np.load(froot + subj + '_' + conds[2] + '.npz')
    itcMean180 = itc_180us['itc_avg']
    
    itc_540us = np.load(froot + subj + '_' + conds[3] + '.npz')
    itcMean540 = itc_540us['itc_avg']
    
    itc_avg = np.load(froot + subj + '_' + conds[4] + '.npz')
    itcMeanAvg = itc_avg['itc_avg']

    itc_avgExd20 = np.load(froot + subj + '_' + conds[5] + '.npz')
    itcMeanAvgExd20 = itc_avgExd20['itc_avg']
    time = itc_avgExd20['t']
    freq = itc_avgExd20['freqs']
    return itcMean20, itcMean60, itcMean180, itcMean540, itcMeanAvg, itcMeanAvgExd20, time, freq

def indexExtraction(t, timeWindow):
    index = np.where(t>timeWindow[0])
    index1 = index[0]
    index1 = index1[0]
    index = np.where(t<timeWindow[1]) 
    index2 = index[0]
    index2 = index2[-1]
    return index1, index2

def noiseFloorEstimate(t, evoked, timeWindow):
    index1, index2 = indexExtraction(t, timeWindow)
    
    noiseFloor = evoked[0:index1].mean(axis=0)
    return noiseFloor, index1, index2

def peak(itc_norm, t, timeWindow):
    index1, index2 = indexExtraction(t, timeWindow)
    
    # Latency from ITD shift to ERP
    index = np.where(t>0.9804)
    indexRef = index[0]
    indexRef = indexRef[0]
    maxITC = max(itc_norm[index1:index2])
    indexMax = np.where(itc_norm == maxITC)
    Lat = t[indexMax] - t[indexRef]
    Lat = round(Lat, 4)
    
    return maxITC, Lat

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
    pl.savefig(froot1 + 'ITCfigures/' + titleStr + '.png')

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

def lowpass(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y
##########################################################################################################################################3
froot = '/media/agudemu/Storage/Data/EEG/ITD/itcCz/'
froot1 = '/media/agudemu/Storage/Data/EEG/ITD/'
stimulus = 'ITD'

wb = xlwt.Workbook()
sheet_itc_all = wb.add_sheet('itc_Cz')
style_bold = xlwt.easyxf('font: bold 1')

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

wb.save(froot1+'/EEG_ITC_Cz.xls')

#subjlist = ['S025', 'S031', 'S046', 'S117', 'S123', 'S127', 'S128', 'S132', 'S133', 'S143', 'S149', 'S051', 'S183', 'S185', 'S187', 'S189', 'S193', 'S195', 'S196', 'S043', 'S072', 'S075', 'S078', 'S135', 'S192', 'S194', 'S197', 'S199', 'S191', 'S084']
subjlist = ['S123']
subjN = 0;
for subj in subjlist:
    subjN = subjN + 1
    conds = ['itc20us', 'itc60us', 'itc180us', 'itc540us', 'itcavg', 'itcavgExd20']
    itcMean20, itcMean60, itcMean180, itcMean540, itcMeanAll, itcMeanAllExd20, t, freq = itcAvgExtraction(conds)
    itcMeanType = [itcMean20, itcMean60, itcMean180, itcMean540, itcMeanAll, itcMeanAllExd20]
    peaksBoth, latBoth = peakITC(itcMeanType, freq, t, conds, [0, 0.3], [1.0, 1.23])
          
    conds = ['itc20us_left', 'itc60us_left', 'itc180us_left', 'itc540us_left', 'itcavg_left', 'itcavgExd20_left']
    itcMean20_left, itcMean60_left, itcMean180_left, itcMean540_left, itcMeanAll_left, itcMeanAllExd20_left, _, _ = itcAvgExtraction(conds)
    itcMeanType = [itcMean20_left, itcMean60_left, itcMean180_left, itcMean540_left, itcMeanAll_left, itcMeanAllExd20_left]
    peaksLeft, latLeft = peakITC(itcMeanType, freq, t, conds, [0, 0.3], [1.0, 1.23])
                
    conds = ['itc20us_right', 'itc60us_right', 'itc180us_right', 'itc540us_right', 'itcavg_right', 'itcavgExd20_right']
    itcMean20_right, itcMean60_right, itcMean180_right, itcMean540_right, itcMeanAll_right, itcMeanAllExd20_right, _, _ = itcAvgExtraction(conds)
    itcMeanType = [itcMean20_right, itcMean60_right, itcMean180_right, itcMean540_right, itcMeanAll_right, itcMeanAllExd20_right]
    peaksRight, latRight = peakITC(itcMeanType, freq, t, conds, [0, 0.3], [1.0, 1.23])
    
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
    wb.save(froot1+'/EEG_ITC_Cz.xls')
