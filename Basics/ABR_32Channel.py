from anlffr.helper import biosemi2mne as bs
import mne
import numpy as np # support for large, multi-dimensional arrays and metrices 
import pylab as pl
import os ### 
import fnmatch ###

# Adding Files and locations
froot = '/Users/baoagudemu1/Desktop/2018Spring/Lab/EEG-Python'

subjlist = ['Hari',]
# Those conditions are for trigger numbers
conds = [[3, 9], [5, 10], [6, 12], [48, 144], [80, 160], [96, 192], [6], [12],
         [96], [192]]

for subj in subjlist:

    fpath = froot + '/' + subj + '/'
    print 'Running Subject', subj 
    rawlist = []
    evelist = []
    
    # Extracting the bdf file names into bdfs
    bdfs = fnmatch.filter(os.listdir(fpath), subj + '_ITD*.bdf')

    if len(bdfs) >= 1:
        for k, bdf in enumerate(bdfs):
            edfname = fpath + bdf
            # Load data and read event channel
            ### Additional channels: needs exploring
            extrachans = [u'GSR1', u'GSR2', u'Erg1', u'Erg2', u'Resp',
                          u'Plet', u'Temp']
            (rawtemp, evestemp) = bs.importbdf(edfname, nchans=36,
                                               extrachans=extrachans) ### rawtemp, evestemp
            rawtemp.set_channel_types({'EXG3': 'eeg', 'EXG4': 'eeg'})
            rawlist += [rawtemp, ]
            evelist += [evestemp, ]
    else:
        RuntimeError('No BDF files found!!')
    
    raw, eves = mne.concatenate_raws(rawlist, events_list=evelist)
    # Filter the data 
    raw.filter(l_freq=1., h_freq=100, picks=np.arange(36))
    tmin, tmax = -0.5, 2.5
#    raw.info['bads'] += ['EXG3', 'EXG4', 'A1', 'A2', 'A30', 'A7', 'A6',
#                         'A24', 'A28', 'A29', 'A3', 'A11', 'A15',
#                         'A16', 'A17', 'A10', 'A21', 'A20', 'A25']
    # Channels on top of head are considered good
#    goods = [28, 3, 30, 26, 4, 25, 7, 31, 22, 9, 8, 21, 11, 12, 18]
    
    abrs = []
    pl.figure()
    
    for cond in conds:
        print 'Doing condition ', cond
        epochs = mne.Epochs(raw, eves, cond, tmin=tmin, proj=False,
                            tmax=tmax, baseline=(-0.5, 0.0),
                            picks=np.arange(36),
                            reject=dict(eeg=100e-6),
                            verbose='WARNING')
        abr = epochs.average()
        abrs += [abr, ]
# Putting all the averages of epochs into one file
mne.write_evokeds(fpath + subj + '_ABR-ave.fif', abrs)

## Plot data
#R = [3, 4, 5]
#L = [0, 1, 2]
#for k in L:
#    abr = abrs[k]
#    x = abr.data * 1e6  # microV
#    t = abr.times * 1e3 - 1.6  # Adjust for delay and use milliseconds
#    ### Referenced to channel 34? mean(axis=0)
#    y = x[goods, :].mean(axis=0) - x[34, :]
#    y = y - y[(t > 0.) & (t < 1.)].mean()
#    pl.plot(t, y, linewidth=2)
#pl.xlabel('Time (ms)', fontsize=14)
#pl.ylabel('ABR (uV)', fontsize=14)
#pl.title('Left Ear', fontsize=14)
#pl.xlim((0., 10.))
#pl.ylim((-1.0, 1.5))
#ax = pl.gca()
#ax.tick_params(labelsize=14)
#pl.legend(['48 dB nHL', '64 dB nHL', '80 dB nHL'])
#pl.legend(['85 dB peSPL', '100 dB peSPL', '115 dB peSPL'])
#pl.show()
#
#pl.figure()
#for k in R:
#    abr = abrs[k]
#    x = abr.data * 1e6  # microV
#    t = abr.times * 1e3 - 1.6  # Adjust for delay and use milliseconds
#    ### Referenced to channel 35?
#    y = x[goods, :].mean(axis=0) - x[35, :]
#    # y = y - y[(t > 0.5) & (t < 1.0)].mean()
#    pl.plot(t, y, linewidth=2)
#pl.xlabel('Time (ms)', fontsize=16)
#pl.ylabel('ABR (uV)', fontsize=16)
#pl.title('Right Ear', fontsize=20)
#pl.xlim((-1.0, 12.))
#pl.ylim((-1., 1.0))
#ax = pl.gca()
#ax.tick_params(labelsize=16)
#pl.legend(['85 dB peSPL', '100 dB peSPL', '115 dB peSPL'])
#pl.show()
