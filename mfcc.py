#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:28:06 2020

@author: Jhonatan Sossa y Juan Pablo Quinchía
"""

import numpy as np


import scipy.fftpack as fft
from scipy.signal import get_window

import matplotlib.pyplot as plt
from sklearn import svm
from funciones import *

#coeficientes cepstrales en las frecuencias de mel
def mfccs_func(sig, fs, audio_framed, FFT_size): 
    '''     recibe la señal normalizada, la frecuencia 
    '''
    #crear ventana Hanning
    window = get_window("hann", FFT_size, fftbins=True) 
    
    #multiplicar ventana de hanning por la señal original
    audio_win = audio_framed * window 
    
    #se aplica la transformada discreta de fourier a cada segmento 
    audio_winT = np.transpose(audio_win)
    audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F') #crear matriz con valores aleatorios
    for n in range(audio_fft.shape[1]): 
         audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]] #tomamos solo la parte positiva
    audio_fft = np.transpose(audio_fft)
    
    #Espectro de potencia de cada ventana
    audio_power = np.square(np.abs(audio_fft)) 
    
    
    #Frecuencias para el espectro
    freq_min = 0
    freq_high = fs/2
    
    #numero de filtros en la escala de mel
    mel_filter_num = 10
    
    #calcular los puntos de los filtros de mel
    filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=44100)
    
    #construccion del filtro
    filters = get_filters(filter_points, FFT_size)
    
    #normalizacion de os filtros segun sus frecuencias de corte
    enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
    filters *= enorm[:, np.newaxis] #se dividen los valores del filtro, por el ancho de cada uno para eliminar el aumento de ruido
    
    #se aplican los filtros de mel a la señal de potencia
    audio_filtered = np.dot(filters, np.transpose(audio_power)) #aplicar cada uno de los filtros a cada una de las ventanas de la señal
    
    #se aplica el logaritmo para dejarlo en esa ecala
    audio_log = 10.0 * np.log10(audio_filtered)
    
    #orden de la transformada discreta del coseno
    dct_filter_num = 13
    
    #se crean los filtros dct
    dct_filters = dct(dct_filter_num, mel_filter_num)
    
    #coeficientes cepstrales
    cepstral_coefficents = np.transpose(np.dot(dct_filters, audio_log))
    
    #derivada de los coeficientes cepstrales
    dcepstral_coefficents = np.diff(cepstral_coefficents,1,axis=0)
    
    #segunda derivada
    ddcepstral_coefficents = np.diff(cepstral_coefficents,2,axis=0)
    
    #se extrae la media, std, kurtosis y asimetría de cada una
    cepstral_coefficents = static_feats(cepstral_coefficents)
    dcepstral_coefficents = static_feats(dcepstral_coefficents)
    ddcepstral_coefficents = static_feats(ddcepstral_coefficents)
    
    #se concatenan
    cepstral_coefficents = np.hstack([cepstral_coefficents,dcepstral_coefficents,ddcepstral_coefficents])
    
    return cepstral_coefficents











