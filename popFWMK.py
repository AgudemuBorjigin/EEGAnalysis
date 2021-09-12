"""
Created on Mon Feb  4 14:37:21 2019

@author: agudemu
"""
import numpy as np # support for large, multi-dimensional arrays and metrices
import pylab as pl
from anlffr.spectral import mtplv
from scipy.io import savemat
from scipy.signal import butter, filtfilt, hilbert  

def highpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def diffSumEvd(evoked_pos, evoked_neg):
    # evoked response analysis
    diffWhole = evoked_pos - evoked_neg
    sumWhole = evoked_pos + evoked_neg
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
    return diffWhole, sumWhole, diffProb, sumProb, diffMasked, sumMasked, diffMasker, sumMasker, diffAdpt, sumAdpt

def summ(epochs_pos, epochs_neg):
    numEvntsPos = epochs_pos.shape
    numEvntsNeg = epochs_neg.shape
    if numEvntsPos[0] > numEvntsNeg[0]:
        summation = epochs_pos[0:numEvntsNeg[0], :, :] + epochs_neg
    else:
        summation = epochs_pos + epochs_neg[0:numEvntsPos[0], :, :]
    return summation, min(numEvntsPos[0], numEvntsNeg[0])

def diff(epochs_pos, epochs_neg):
    numEvntsPos = epochs_pos.shape
    numEvntsNeg = epochs_neg.shape
    if numEvntsPos[0] > numEvntsNeg[0]:
        subt = epochs_pos[0:numEvntsNeg[0], :, :] - epochs_neg
    else:
        subt = epochs_pos - epochs_neg[0:numEvntsPos[0], :, :]
    return subt

def freqAnsis(sig):
    sig_fft = np.fft.fft(sig)
    magSig = np.abs(sig_fft)
    phase = np.angle(sig_fft)
    return magSig, phase

def fftPlot(evoked, fs, chop, chop2, stimType, color, yLabel):
    # frequency analysis
    t = np.arange(0, len(evoked)/float(fs), 1/float(fs))
    mag, phase500 = freqAnsis(evoked[np.logical_and(t > chop, t < chop2)])
    y = evoked[np.logical_and(t > chop, t < chop2)]
    x = t[np.logical_and(t > chop, t < chop2)]
#    y = evoked
    
    freq = np.linspace(0, fs/4, len(mag)/4)
#    fig = pl.figure(figsize = (20, 5))
    fig = pl.figure()
#    ax = fig.add_subplot(211)
#    titleStr = subj + '_' + stimType
#    pl.title(titleStr, fontsize=14)
#    pl.xlabel('Time (s)', fontsize=14)
#    pl.ylabel('Evoked response', fontsize=14)
#    ax.plot(x, y)
    

    ax = fig.add_subplot(111)
    peak = max(mag[0:len(mag)/4])
    index = np.where(mag[0:len(mag)/4] == peak)
    pl.plot(freq, mag[0:len(mag)/4]*1e5, color, linewidth = 2)
#    ax.plot(freq[index], peak, 'r+', markersize = 12, linewidth = 8)
#    ax.annotate(repr(peak), xy = (freq[index], peak), xytext = (freq[index], peak))
    #pl.xlabel('Frequency (Hz)', fontsize = 45)
    pl.ylabel(yLabel, fontsize = 45)
    #pl.ylim(-1, 10)
    pl.xlim(0, 2000)
    ax.set_xticks([0, 500, 1000, 2000])
    ax.set_xticklabels(['0', '500', '1000', '2000'])
    ax.tick_params(labelsize=45)
#    pl.savefig(froot + '/FWMKfigures_FCz/' + titleStr + '.png')
    return peak, index

def fftPlot_avg(mag, fs, color, yLabel):
    freq = np.linspace(0, fs/4, len(mag)/4)
    fig = pl.figure()
    ax = fig.add_subplot(111)
    pl.plot(freq, mag[0:len(mag)/4]*1e5, color, linewidth = 2)
    pl.ylabel(yLabel, fontsize = 45)
    pl.ylim(-1, 5)
    pl.xlim(0, 2000)
    ax.set_xticks([0, 500, 1000, 2000])
    ax.set_xticklabels(['0', '500', '1000', '2000'])
    ax.tick_params(labelsize=45)
    return 

def plot_plv(f, plv, f_500, plv_500, mean, std):
    pl.figure()
    pl.plot(f, plv)
    pl.plot(f_500, plv_500, 'r+', markersize = 12, linewidth = 8)
    pl.axhline(y = mean + 2 * std, color = 'k', linestyle = 'dashed')
    pl.axhline(y = mean, color = 'k')
    pl.text(0, mean, str(mean))
    pl.axhline(y = mean - 2 * std, color = 'k', linestyle = 'dashed')
    pl.xlabel('Frequency [Hz]')
    pl.ylabel('PLV')
    pl.title('PLV:' + subj)

def plvAnalysis(epochs, Type):
    # phase locking analysis
    params = dict(Fs = 16384, tapers = [1, 1], fpass = [400, 600], itc = 0)
    # plv to TFS
    epochs = np.transpose(epochs, (1, 0, 2)) # switching the first and second columns
    plv, f = mtplv(epochs, params)
    index = np.where(plv[31,] == max(plv[31,])) # CHANGE AS NEEDED: 32st channel (Cz)
    plv_32 = plv[31,]
    plv_32_max = plv_32[index]
    f_max = f[index]
    dimen = epochs.shape
    numtrials = dimen[1]
    # saving plvs
    dictMat = {"plv_32": plv_32_max, "f": f_max, "Fs": fs, "trialNum": numtrials}
    savemat('/media/agudemu/Storage/Data/EEG/FFR/plvs/'+subj+Type, dictMat)
    # collecting individual plvs into array for all subjects
    return plv_32_max, f_max, numtrials

####################################################################################################################################
froot = '/media/agudemu/Storage/Data/EEG/FFR/epochs_evoked_FCz/'

subjs = ['S025', 'S031', 'S043', 'S051', 'S072', 'S078', 'S075', 'S084', 'S117', 'S123', 'S127', 
         'S128', 'S132', 'S133', 'S149', 'S183', 'S185', 'S187', 'S191', 'S194', 'S195', 'S196', 'S197', 
         'S216', 'S218']
#subjs = ['S084'] # median subject
#subjs = ['S185'] # 2nd best subject
refs = np.zeros(len(subjs))
fs = 16384
chop = 0e-3
chop2 = 50e-3

for k, subj in enumerate(subjs):
    data = np.load(froot + subj + '.npz')
#    epochs_masker_pos = data['data_masker_pos']
#    epochs_masker_neg = data['data_masker_neg']
#    epochs_prob_pos = data['data_prob_pos']
#    epochs_prob_neg = data['data_prob_neg']
#    epochs_adpt_diff = data['data_adpt_diff']
    evoked_pos = data['evoked_pos']
    evoked_neg = data['evoked_neg']
#    diffWhole, sumWhole, diffProb, sumProb, diffMasked, sumMasked, diffMasker, sumMasker, diffAdpt, sumAdpt = diffSumEvd(evoked_pos, evoked_neg)
#    PosMasker = evoked_pos[int(0.152*fs):int(0.159*fs)]
#    NegMasker = evoked_neg[int(0.152*fs):int(0.159*fs)]
#    masker = PosMasker + NegMasker
#    refs[k] = np.sqrt(np.mean(np.multiply(masker, masker)))
#dictDataArray = {"refs": refs}
#savemat('rms', dictDataArray)  
#    ref_rms = np.sqrt(np.mean((masker^2)))
#    peak_abr_pos, _ = fftPlot(PosMasker, fs, 0, 12e-3, 'ABR to positive masker')
#    peak_abr_neg, _ = fftPlot(NegMasker, fs, 0, 12e-3, 'ABR to negative masker')
#    peak_masker_sum, _ = fftPlot(sumMasker, fs, chop, chop2, 'MaskerSum')
#    peak_adpt, _ = fftPlot(diffAdpt, fs, chop, chop2, 'AdptDiff')
#    peak_masker_diff, _ = fftPlot(diffMasker, fs, chop, chop2, 'MaskerDiff')
#    peak_prob_diff, _ = fftPlot(diffProb, fs, chop, chop2, 'ProbDiff')
#    
#    data_masker_sum, _ = summ(epochs_masker_pos, epochs_masker_neg)
#    plv_masker_sum, f_masker_sum, _ = plvAnalysis(data_masker_sum, 'MaskerSum')
#    plv_adpt_diff, f_adpt_diff, _ = plvAnalysis(epochs_adpt_diff, 'AdptDiff')
#    data_masker_diff = diff(epochs_masker_pos, epochs_masker_neg)
#    plv_masker_diff, f_masker_diff, _ = plvAnalysis(data_masker_diff, 'MaskerDiff')
#    data_prob_diff = diff(epochs_prob_pos, epochs_prob_neg)
#    plv_prob_diff, f_prob_diff, _ = plvAnalysis(data_prob_diff, 'ProbDiff')
    
    # initialization for grand average across population
    diffWhole, sumWhole, diffProb, sumProb, diffMasked, sumMasked, diffMasker, sumMasker, diffAdpt, sumAdpt = diffSumEvd(evoked_pos, evoked_neg)
    t = np.arange(0, len(diffAdpt)/float(fs), 1/float(fs))
    mag_diffAdpt, phase500 = freqAnsis(diffAdpt[np.logical_and(t > chop, t < chop2)])
    t = np.arange(0, len(diffProb)/float(fs), 1/float(fs))
    mag_diffProb, phase500 = freqAnsis(diffProb[np.logical_and(t > chop, t < chop2)])
    t = np.arange(0, len(sumAdpt)/float(fs), 1/float(fs))
    mag_sumAdpt, phase500 = freqAnsis(sumAdpt[np.logical_and(t > chop, t < chop2)])
    t = np.arange(0, len(sumProb)/float(fs), 1/float(fs))
    mag_sumProb, phase500 = freqAnsis(sumProb[np.logical_and(t > chop, t < chop2)])
    if k == 0:
        length = len(evoked_pos)
        length_fft = len(mag_diffAdpt)
        evoked_poss = np.zeros((len(subjs), length))
        evoked_negs = np.zeros((len(subjs), length))
        fft_diff_adpt = np.zeros((len(subjs), length_fft))
        fft_sum_adpt = np.zeros((len(subjs), length_fft))
        fft_diff_prob = np.zeros((len(subjs), length_fft))
        fft_sum_prob = np.zeros((len(subjs), length_fft))
    evoked_poss[k] = evoked_pos
    evoked_negs[k] = evoked_neg
    fft_diff_adpt[k] = mag_diffAdpt
    fft_sum_adpt[k] = mag_sumAdpt
    fft_diff_prob[k] = mag_diffProb
    fft_sum_prob[k] = mag_sumProb

# time domain population response
evoked_pos_mean = np.mean(evoked_poss, axis = 0)
evoked_neg_mean = np.mean(evoked_negs, axis = 0)
diffWhole, sumWhole, diffProb, sumProb, diffMasked, sumMasked, diffMasker, sumMasker, diffAdpt, sumAdpt = diffSumEvd(evoked_pos_mean, evoked_neg_mean)    
t = np.arange(0, len(diffProb)/float(fs), 1/float(fs))
lineWidth = 2
fontSize = 45
# plot
fig = pl.figure()
pl.subplot(2, 1, 1)
t = np.arange(0, len(diffWhole)/float(fs), 1/float(fs))
pl.plot(t, diffWhole*1e7, 'g', linewidth = lineWidth)
pl.tick_params(labelsize=fontSize, bottom=False, labelbottom=False)
pl.subplot(2, 1, 2)
t = np.arange(0, len(sumWhole)/float(fs), 1/float(fs))
pl.plot(t, sumWhole*1e7, 'b', linewidth = lineWidth)
pl.tick_params(labelsize=fontSize)
#pl.xlabel('Time [s]', fontsize = fontSize)
# commom y label
fig.text(0.06, 0.5, 'Voltage [0.1 uV]', ha='center', va='center', rotation='vertical', fontsize = fontSize)
pl.ylim(-5, 5)

#t_sine = np.arange(0, len(diffAdpt)/float(fs), 1/float(fs))
#amplitude = np.sin(2*np.pi*1000*t_sine)
#pl.figure()
#pl.plot(t_sine, amplitude)
#mag, phase = freqAnsis(amplitude)
#freq = np.linspace(0, fs/4, len(mag)/4)
#pl.figure()
#pl.plot(freq, mag[0:len(mag)/4])

# average of magnitudes in frequency domain; took fft of each individual then took the average of all magnitudes
fft_diff_adpt_mean = np.mean(fft_diff_adpt, axis = 0)
fft_sum_adpt_mean = np.mean(fft_sum_adpt, axis = 0)
fft_diff_prob_mean = np.mean(fft_diff_prob, axis = 0)
fft_sum_prob_mean = np.mean(fft_sum_prob, axis = 0)

fftPlot_avg(fft_diff_adpt_mean, fs, 'g', '')
fftPlot_avg(fft_diff_prob_mean, fs, 'g', 'Amplitude [1 nV]')
fftPlot_avg(fft_sum_adpt_mean, fs, 'b', '')
fftPlot_avg(fft_sum_prob_mean, fs, 'b', 'Amplitude [1 nV]')
#peak_masker, frequency_masker = fftPlot(sumMasker, fs, chop, chop2, 'The sum of responses to positive and negative polarities')
#peak_adpt, frequency_adpt = fftPlot(diffAdpt, fs, chop, chop2, 'DiffAdpt')
#plv_32_max, f_max, numtrials, plv, f = plvAnalysis(epochs_adpt_diff, [400, 700])
    
#t = np.arange(0, len(evoked_pos)/float(fs), 1/float(fs))
#evoked_pos_mean = np.mean(evoked_poss, axis = 0)
#evoked_neg_mean = np.mean(evoked_negs, axis = 0)
#diffAdpt_mean = np.mean(diffAdpts, axis = 0)
#sumMasker_mean = highpass_filter(np.mean(sumMaskers, axis=0), 800, fs, 4)
#evoked_sum = evoked_pos_mean + evoked_neg_mean
#evoked_diff = evoked_pos_mean - evoked_neg_mean
#fontSize = 45
#lineWidth = 2
#pl.subplot(2, 1, 1)
#pl.plot(t, evoked_sum, 'g', linewidth = lineWidth, label = 'sum')
#pl.legend()
#pl.subplot(2, 1, 2)
#pl.plot(t, evoked_diff, 'r', linewidth = lineWidth, label = 'diff')
#pl.legend()