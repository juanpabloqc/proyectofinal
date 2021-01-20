#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dic  3 13:01:52 2020

@author: Jhonatan Sossa y Juan Pablo Quinchía
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn import  svm



from scipy import stats




def zscore(X):
    #Zscore
    X= stats.zscore(X, axis=0)
    X=X[:,~np.isnan(X).any(axis=0)]
    return X

#Evaluacion de rendimiento
def EvalRend(LR, LC, LC1, LC2):
    
    C1=np.where(LR==LC1)
    C2=np.where(LR==LC2)
    
        
    TP=float(len(np.where(LC[C1]==LC1)[0]))
    FN=float(len(np.where(LC[C2]==LC1)[0])) #Considero que acá hay un error en el tutorial
    FP=float(len(np.where(LC[C1]==LC2)[0])) #Aquí también PREGUNTAR
    TN=float(len(np.where(LC[C2]==LC2)[0]))
    
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    
        #Porcentaje de acierto
    A=float(TP + TN)/float(TP + FN + FP + TN)*100
        #Sensibilidad
    S=float(TP)/float(TP + FN)*100
        
        #Especificidad
    E=float(TN)/float(TN + FP)*100
        
    CMatriz = np.array([[TN, FP],[FN, TP]])

    return A, S, E

def EvalRendMC(y_test_, y_pred_):
    
    CMatriz = confusion_matrix(y_test_, y_pred_)
    
    rend_params = []
    for i in range(CMatriz.shape[0]):
        suma=0
        for j in range(CMatriz.shape[0]):
            suma+=float(CMatriz[i,j])
            
        rend_params.append(float(CMatriz[i,i])/suma*100)
    
    A = float(np.trace(CMatriz))/float(np.sum(CMatriz))*100
    
    return A, rend_params, CMatriz

def folds_dist(spkID,y,k_splits=10):
   """
   Fold distribution
   """
   #Validacion cruzada de k-particiones
#    kf = KFold(k_splits,shuffle=True)
   kf = StratifiedKFold(n_splits=k_splits, random_state=None, shuffle=False)
   idxtrain = []
   idxtest = []
   for x,y in kf.split(spkID,y):
       idxtrain.append(x)
       idxtest.append(y)
   return idxtrain,idxtest

def train_svc(X_train,y_train,ksplits):  
#   Set the parameters by cross-validation
   tuned_parameters = [{'kernel': ['rbf'],
                        'gamma': [0.001],
                        'C': [10]}] 
    
   clf = GridSearchCV(svm.SVC(class_weight='balanced'),
                          tuned_parameters,cv=ksplits)
   
   
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=59)

par=[]
M=[]
y_array =[]

fpr_svm_ = []
tpr_svm_ = []

#Estandarización de la matriz de entrenamiento
mXtrain = np.mean(X_train,axis= 0)
sXtrain = np.std(X_train,axis= 0)    
X_train = (X_train-mXtrain)/sXtrain

#Estandarización de la matriz de test
X_test = (X_test-mXtrain)/sXtrain

   
 
#Ingresa a la función de entrenamientp
clf = svm.SVC(class_weight='balanced', gamma= 0.001, C=10)
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)

y_array.append((y_test,y_pred))

#Confusion Matrix
A, params, Matrix = EvalRendMC(y_test, y_pred) #MC


f_score= f1_score(y_test, y_pred, average='macro')

print(classification_report(y_test,y_pred))

par.append(np.hstack([A, params, f_score, 0.001, 10]))




