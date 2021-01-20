#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 03 23:28:55 2020

@author: Jhonatan Sossa y Juan Pablo Quinchía
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
import tkinter


from scipy.io.wavfile import read,write 
import os
from IPython.display import Audio
import simpleaudio as sa
from PIL import Image, ImageTk
from main import classifier
from funciones import normalize_audio


class mclass:
    def __init__(self, window, list_wavs):
        self.window = window
        self.ban = True
        
        
        self.fig = Figure(figsize=(5,2))
        self.a = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack()
        
        self.var = IntVar(window)
        for val in range(len(list_wavs)):
             Radiobutton(window, 
                          text=list_wavs[val],
                          padx = 20, 
                          variable=self.var, 
                          command=self.sel,
                          value=val).pack(fill = 'x' , expand = 1)#anchor=tk.W)

        self.label = Label(window)
        self.label.pack()
        
        
        self.button = tkinter.Button (window, text="Escoger", command=self.aceptar)
        self.button.pack()
        self.button1 = tkinter.Button(master=window, text="Salir", command=self._quit)
        self.button1.pack(side=tkinter.BOTTOM)


    def aceptar(self):
        cancion=self.var.get()
        print(list_wavs[cancion])
    
        file_audio=(path_files+'/'+list_wavs[cancion])
        fs, sig=read(file_audio)
        sig = normalize_audio(sig)
        t=np.arange(0, float(len(sig))/fs, 1.0/fs)
        
        self.a.clear()
        
        self.a.plot(t, sig,color='blue')
    
        self.a.set_title (list_wavs[cancion], fontsize=8)
        self.a.set_ylabel("Amplitud", fontsize=6)
        self.a.set_xlabel("Tiempo", fontsize=6)
        
        self.canvas.draw()
        
        wave_obj = sa.WaveObject.from_wave_file(path_files+'/'+list_wavs[cancion])
        play_obj = wave_obj.play()
        play_obj.wait_done()
        
        instrument = classifier(sig, fs)
        
        result = "El instrumento es un " + instrument
        self.label.config(text = result)
        
    def sel(self):
       selection = "Seleccionaste la opción " + str(self.var.get() + 1)
       print(selection)
       
        
        
    def _quit(self):
        self.window.quit()     
        self.window.destroy()  


path_files = './temitas/'
list_wavs = os.listdir(path_files)

listofzeros = [0] * len(list_wavs)
for i in range(len(list_wavs)):
    listofzeros[int(re.findall('\d+', list_wavs[i] )[0])-1] = list_wavs[i]
    
list_wavs = listofzeros

window= Tk()
start= mclass (window, list_wavs)
window.mainloop()