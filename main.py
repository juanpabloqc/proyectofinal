# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:11:53 2020

@author: Jhonatan Sossa y Juan Pablo Quinchía
"""


import os
import sys
import numpy as np
from mfcc import mfccs_func
from funciones import normalize_audio, frame_audio
from scipy.io.wavfile import read,write 
from f0_func import f0_func
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc,classification_report
from sklearn import neighbors, datasets, svm, metrics

#Extrae las características de las señales a clasificar y las clasifica
def classifier(sig, fs):
    '''recibe la señal normalizada y la frecuencia de muestreo
    '''
    name = {0:"violin", 1:"piano", 2:"saxofon", 3:"guitarra electrica"}
    
    #Carga el modelo
    clf = pickle.load(open('finalized_model.sav', 'rb'))
    
    #carga la media y la desviación estándar de la matriz de entrenamiento
    mde = np.loadtxt('data.txt')
    
    feat_mat=[]
    
    #se segmenta la señal
    FFT_size=2048
    hop_size=10
    audio_framed = frame_audio(sig, FFT_size=FFT_size, hop_size=hop_size, fs=fs) #divinir el audio en ventanas
    
    #se genera el vector de características (159 en total)
    features = mfccs_func(sig, fs, audio_framed, FFT_size)
    f0 = f0_func(sig=sig, fs=fs, audio_framed=audio_framed)
    feat_mat.append(np.hstack([f0, features]))
    
    #Se estandariza la matriz de características a clasificar con la media y la desviación estándar de la matriz de entrenamiento
    feat_mat = (feat_mat-mde[0])/mde[1]
    
    #se hace la clasificación
    y_pred = clf.predict(feat_mat)
    
    #devuelve el nombre del instrumento
    return name[int(y_pred)]

