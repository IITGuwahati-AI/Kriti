# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import scipy.io.wavfile
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.fftpack import dct
sample_rate, signal = scipy.io.wavfile.read('/Users/Downloads/AUD-20190525-WA0006.wav')  
pre_emphasis = 0.97
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
frame_size = 0.025
frame_stride = 0.01
frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  
pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(emphasized_signal, z) 
indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]
frames *= np.hamming(frame_length)
NFFT = 512
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
plt.plot(pow_frames)
nfilt = 26
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  
hz_points = (700 * (10**(mel_points / 2595) - 1))
bin = np.floor((NFFT + 1) * hz_points / sample_rate)
fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * np.log10(filter_banks)
num_ceps = 13
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
(nframes, ncoeff) = mfcc.shape
n = np.arange(ncoeff)
cep_lifter = 22
lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
mfcc *= lift 
mfcc2 = mfcc
mfcc2 -= (np.mean(mfcc, axis=0) + 1e-8)


mfcc_data1= np.swapaxes(mfcc1, 0 ,1)
cax = ax.imshow(mfcc_data1, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
mfcc_data2= np.swapaxes(mfcc2, 0 ,1)
cax = ax.imshow(mfcc_data2, interpolation='nearest', cmap=cm.coolwarm, origin='lower')


fig = plt.figure()
ax1 = plt.subplot(211)
plt.plot(mfcc_data1)
ax2 = plt.subplot(212, sharex = ax1)
plt.plot(mfcc_data2)

#########################################

from dtw import dtw
from numpy.linalg import norm
dist, cost, acc_cost, path = dtw(mfcc1, mfcc2, dist=lambda x, y: norm(x - y, ord=1))
dist
plt.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.xlim((-0.5, cost.shape[0]-0.5))
plt.ylim((-0.5, cost.shape[1]-0.5))















