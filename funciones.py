#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:39:34 2020

@author: Jhonatan Sossa y Juan Pablo Quinchía
"""

import scipy
from scipy.io import wavfile
import IPython.display as ipd
from scipy.stats import kurtosis, skew
import numpy as np


#Normaliza la señal
def normalize_audio(sig):
    '''Recibe la señal 
    '''
    sig = np.mean(sig,axis=1)
    sig=sig-np.mean(sig)
    sig = sig / np.max(np.abs(sig))
    return sig

#Envenana la señal
def frame_audio(sig, FFT_size, hop_size, fs):
    '''Recibe la señal normalizada, el tamaño de la FFT, el tiempo de duraación de la ventana y la frecuencia de muestreo
    '''
    sig = np.pad(sig, int(FFT_size / 2), mode='reflect')
    frame_len = np.round(fs * hop_size / 1000).astype(int)
    frame_num = int((len(sig) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num,FFT_size))
    
    for n in range(frame_num):
        frames[n] = sig[n*frame_len:n*frame_len+FFT_size]
    
    return frames

#Pasa de frecuencia a mel
def freq_to_mel(freq):
    '''Recibe frecuencia
    '''
    return 2595.0 * np.log10(1.0 + freq / 700.0)

#Pasa de mel a frecuencia
def met_to_freq(mels):
    '''Recibe mels
    '''
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

#Se obtienen las frecuencias de corte para los filtros de mel
def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    '''Recibe frecuencia mínima, frecuencia máxima, número de filtros de mel, tamaño de la FFT y la frecuencia de muestreo
    '''
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)

    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = met_to_freq(mels)
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs #frecuencias de corte

#Se general los filtros de mel
def get_filters(filter_points, FFT_size):
    '''Recibe las frecuencias de corte de los fitros y el tamaño de la FFT
    '''
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters


#Hace la transformada discreta del coseno
def dct(dct_filter_num, filter_len):
    ''' recibe número de filtros dct y la longitud
    '''
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis



def static_feats(featmat):
    """Compute static features
    :param featmat: Feature matrix
    :returns: statmat: Static feature matrix
    """
    mu = np.mean(featmat,0)
    st = np.std(featmat,0)
    ku = kurtosis(featmat,0) #mide qué tan achatada está la campana de Gauss. CUando son negativos, está muy esparcida entonces es muy achatada 
    sk = skew(featmat,0)    #asimetría (hacia qué lado setán centrados los datos)
    statmat = np.hstack([mu,st,ku,sk])    
    return statmat








