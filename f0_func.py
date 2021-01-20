# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:05:50 2020

@author: Jhonatan Sossa y Juan Pablo Quinchía
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read,write #Leer y guardar audios
from scipy.signal import butter, lfilter, find_peaks
import scipy
from IPython.display import Audio
from funciones import normalize_audio, frame_audio, static_feats
from scipy.signal import find_peaks


#Función para extraer semitonos
def Hz2semitones(freq): #author: J. C. Vasquez-Correa
    A4=440.
    C0=A4*2**(-4.75)
    if freq>0:
        h=12*np.log2(freq/C0)
        octave=h//12.
        return h+octave
    else:
        return C0

#Función para extrar la frecuencia fundamental
def f0_func(sig, fs, audio_framed):
    '''Recibe la señal normalizada, la frecuencia fundamental y la señal enventanada
    '''
    f0=[]
    
    E=[]
    
    #Se genera un vector con la 
    for frame in audio_framed:
        xcorr = np.correlate(frame, frame, 'full')
        E.append(xcorr[np.where(xcorr == np.max(xcorr))[0][0]])
    
    #promedio de la energía de las ventanas
    prom_E=np.sum(E)/len(E)
      
    #Se calcula la distancia entre el valor máximo de la autocorrelación y el segundo pico más alto para encontrar la frecuencia fundamental
    for frame in audio_framed:
        xcorr = np.correlate(frame, frame, 'full')
        tao = np.arange(-len(frame)+1, len(frame), 1)
    
        tao = tao[np.where(xcorr == np.max(xcorr))[0][0]:len(tao)]
        xcorr = xcorr[np.where(xcorr == np.max(xcorr))[0][0]:len(xcorr)]
        peaks, _ = find_peaks(xcorr, height=0)
        peaks_amplitude = xcorr[peaks]
        
        #Se seleccionan sólo las ventanas que tengan energía por encima del promedio
        if xcorr[0]>1*prom_E and len(peaks)>0:
            
            lag= peaks[np.where(peaks_amplitude == np.max(peaks_amplitude))][0]
            f0.append(fs/lag)
    
    #Se concatena verticalmente la frecuencia fundamental de todas las ventanas seleccionadas
    f0 = np.vstack(f0)
    
    #Se extrae la media, desviación estándar y el semitono
    f0mu = np.mean(f0[f0!=0])
    f0std = np.std(f0[f0!=0])
    f0varsemi=Hz2semitones(f0std**2)
    return np.hstack([f0mu, f0std, f0varsemi])
 
 















