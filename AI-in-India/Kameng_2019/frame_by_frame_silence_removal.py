#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:23:37 2019

@author: uchiha_ashish
"""
import numpy as np
import scipy.io.wavfile
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import python_speech_features
from scipy import io
sample_rate, signal = scipy.io.wavfile.read('/Users/uchiha_ashish/Downloads/AUD-20190525-WA0006.wav')  
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
m = frames.max(axis=1)
silent_frames = np.array(np.where(frames.max(axis=1) < 500))
frames_final = np.delete(frames, silent_frames[:, :], axis=0)
signal_final2 = python_speech_features.sigproc.deframesig(frames_final, len(emphasized_signal), frame_length, frame_step)
plt.plot(signal_final2)



import numpy as np
import scipy.io.wavfile
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import python_speech_features
from scipy import io
sample_rate, signal = scipy.io.wavfile.read('/Users/uchiha_ashish/Downloads/AUD-20190525-WA0005.wav')  
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
m = frames.max(axis=1)
silent_frames = np.array(np.where(frames.max(axis=1) < 700))
frames_final = np.delete(frames, silent_frames[:, :], axis=0)
signal_final1 = python_speech_features.sigproc.deframesig(frames_final, len(emphasized_signal), frame_length, frame_step)
plt.plot(signal_final1)


fig = plt.figure(figsize=(30, 4))
ax1 = plt.subplot(211)
plt.plot(signal_final1)
ax2 = plt.subplot(212, sharex = ax1)
plt.plot(signal_final2)

plt.plot(signal_final1)
plt.plot(signal_final2)
 
