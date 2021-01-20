#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:01:52 2020

@author: Jhonatan Sossa y Juan Pablo Quinchía
"""

import os
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc,classification_report
from sklearn import neighbors, datasets, svm, metrics
from sklearn.decomposition import PCA


from scipy.io.wavfile import read,write
from scipy import stats


#Función de entrenamiento
def train_svc(X_train,y_train):  
    #Parámetros para el entrenamiento de la SVM
   tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001], 'C': [10]}] 

   clf = GridSearchCV(svm.SVC(class_weight='balanced'),
                          tuned_parameters)
   
   #Entrenamiento
   clf.fit(X_train, y_train)
   return clf

path_base = os.path.dirname(os.path.abspath(__file__))

#Lee los archivos con las características
feats_vio = np.loadtxt("mfcc_features_vio_h2z.txt")
feats_pia = np.loadtxt("mfcc_features_pia_h2z.txt")
feats_sax = np.loadtxt("mfcc_features_sax_h2z.txt")
feats_gac = np.loadtxt("mfcc_features_gac_h2z.txt")

#Concatenación de las características
X =[feats_vio, feats_pia, feats_sax, feats_gac]

#Concatenación vertical de las características
X = np.vstack(X)

#Se asigna una etiqueta acada característica (0:violin, 1:piano, 2:saxofon, 3:guitarra electrica)
y=np.ones(len(X))
for i in range(len(X)):
    if i<len(X)/4:
        y[i] = 0
    elif i>=len(X)/4 and i<2*len(X)/4:
        y[i] = 1
    elif i>=2*len(X)/4 and i<3*len(X)/4:
        y[i] = 2
    else:
        y[i] = 3


#Matriz de características para el entrenamiento y sus respectivas etiquetas
X_train = X
y_train = y

#Estandarización de la matriz de entrenamiento
mXtrain = np.mean(X_train,axis= 0)
sXtrain = np.std(X_train,axis= 0)    
X_train = (X_train-mXtrain)/sXtrain

#Ingresa a la función de entrenamientp
clf = train_svc(X_train,y_train)

# Guarda el modelo como 'finalized_model.sav'
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

#Guarda la media y la desviación estándar de la matriz de entrenamiento
np.savetxt('data.txt',np.vstack([mXtrain,sXtrain]))




