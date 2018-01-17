#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:55:05 2018

@author: baoagudemu1
"""

import pylab as pl
import numpy as np
import scipy as sp 
# Extra "import" statement other than the one above is required?
from scipy.signal import butter, lfilter 
import os

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b, a = butter(order,[low, high], btype = 'band')
    return b, a 

def butter_bandpass_filter(data, lowcut, highcut, fs, order = 5): #order = 5 expression?
    b, a = butter_bandpass(lowcut, highcut, fs, order = order)
    y = lfilter(b, a, data)
    return y
    

fs = 44100;

t = np.linspace(0, 1, fs*1)
noise = np.random.normal(0, 1, t.shape)

noise_filtered = butter_bandpass_filter(noise, 1000, 2000, fs, order = 5)

pl.plot(t, noise)
pl.plot(t, noise_filtered)
pl.show()

