#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:38:57 2020

@author: Jhonatan Sossa y Juan Pablo Quinchía
"""
import os
import sys
import numpy as np
from mfcc import mfccs_func
from funciones import normalize_audio, frame_audio
from scipy.io.wavfile import read,write 
from scipy.stats import kurtosis, skew
from f0_func import f0_func

path_base = os.path.dirname(os.path.abspath(__file__))


#Aquí se extraen las características para cada instrumento (violín, saxofón, piano y guitarra eléctrica)

#vio, pia, sax y gac
group = 'vio'
print(group)
path_files = './IRMAS-TrainingData/'+group

#Se cargan todos los audios que se encuentren en la carpeta
list_wavs = os.listdir(path_files)
feat_mat = []

#Se recorre cada uno de los audios
for idx_wav in list_wavs:
    
    fs,sig = read(path_files+'/'+idx_wav)
    
    #normaliza la señal
    sig = normalize_audio(sig)
    
    #se segmenta la señal
    FFT_size=2048
    hop_size=10
    audio_framed = frame_audio(sig, FFT_size=FFT_size, hop_size=hop_size, fs=fs) #divinir el audio en ventanas
    
    #se extraen los MFCCs y la frecuencia fundamental
    features = mfccs_func(sig, fs, audio_framed, FFT_size)
    f0 = f0_func(sig=sig, fs=fs, audio_framed=audio_framed)
    
    #se concatenan las características (en total son 159 características por audio)
    feat_mat.append(np.hstack([f0, features]))

#Se convierte la lista en una matriz de características de numpy
feat_mat = np.vstack(feat_mat)

#Se guardan las características como un txt
np.savetxt(path_base+'/mfcc_features_'+group+'_own.txt',feat_mat)
print('!'*50)
print('!'*50)
print('saved in', path_base+'/mfcc_features_'+group+'_own.txt')

























"""
    feats = []
    frames = extract_windows(sig,int(0.02*fs),int(0.01*fs))
    for frame in frames:
        #Mfcc
        fmfcc = python_speech_features.mfcc(frame,fs,nfft=2048)
        feats.append(fmfcc)

    if len(sig)!=0:
        feats = np.vstack(feats)
    else:
        feats = np.asarray(feats)
    dfeats = np.diff(feats,1,axis=0)
    ddfeats = np.diff(feats,2,axis=0)
    
    #Compute statistics
    feats = static_feats(feats)
    dfeats = static_feats(dfeats)
    ddfeats = static_feats(ddfeats)
    """






