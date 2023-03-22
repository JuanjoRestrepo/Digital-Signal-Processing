import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from matplotlib import image
import pandas as pd
import scipy as sp
import scipy.io
import math
import cv2
import random
from PIL import Image
import time
import statsmodels.api as sm


y, sr = librosa.load('ElCoronel.wav', sr=16000)
print("Current audio sampling rate: ", sr)

print("Audio Duration:", librosa.get_duration(y=y, sr=sr))

D = librosa.stft(y, hop_length=64, win_length=256)  # STFT of y
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)


librosa.display.waveshow(y, sr=sr)
plt.savefig('plot1.png')
plt.show()
spec = librosa.display.specshow(S_db, sr=sr,hop_length=64, x_axis='s', y_axis='linear')
plt.savefig('plot2.png')

plt.show()

#288 x 432
im1 = Image.open('plot2.png')
im2 = Image.open('plot2.png')
#ESPECTOGRAMA AUDIO
im_crop = im1.crop((55, 35, 390, 253))
im_crop.save('plot31.png', quality=95)

im_crop=im_crop.convert('L')

im_crop.save('plot3.png', quality=95)
#ESPECTOGRAMA FONEMA
#SON 335 CUADROS DEL ESPECTOGRAMA ENTERO
#SON 9,76 SEGUNDOS
#DE 1,6 s A 2,5 se dice coronel aproximadamente
#LA MAYOR POTENCIA DE LA FRECUENCIA SE VE ENTRE 0 Y 3000, ES HASTA 8000 Hz
#SON 218 CUADROS DE ALTO DEL ESPECTROGRAMA
#SE RECORTA ENTONCES EN HORIZONTAL DEL CUADRO 58 AL CUADRO 86
#SE RECORTA ENTONCES EN HORIZONTAL DEL CUADRO 136 AL CUADRO 218
#AJUSTADO SE SUMA 55 EN HORIZONTAL Y 35 EN VERTICAL
#103 a 141.    171 a 253
im_crop = im2.crop((100, 203, 150, 253))
im_crop.save('plot41.png', quality=95)
im_crop = im_crop.convert("L")
im_crop.save('plot4.png', quality=95)

print("CARACTERISTICAS IMAGEN SPECTOGRAMA ORIGINAL")

original= image.imread('plot2.png')
print(original.dtype)
print(original.shape)

print("CARACTERISTICAS IMAGEN SPECTOGRAMA AUDIO GRISES")

spec_audio = image.imread('plot3.png')
spec_audio1 = image.imread('plot31.png')

print(spec_audio.dtype)
print(spec_audio.shape)
plt.imshow(spec_audio)
plt.show()

print("CARACTERISTICAS IMAGEN SPECTOGRAMA FONEMA GRISES")

spec_fonema = image.imread('plot4.png')
spec_fonema1 = image.imread('plot41.png')
print(spec_fonema.dtype)
print(spec_fonema.shape)
'''

plt.imshow(spec_fonema)
plt.show()

'''

#calculate cross correlation
'''
cor= sm.tsa.stattools.ccf(spec_fonema, spec_audio)
plt.imshow(cor)
plt.show()

#calculate cross correlation

cor= scipy.signal.correlate2d(spec_audio, spec_fonema, mode='full', boundary='fill', fillvalue=0)
plt.imshow(cor)
plt.show()


cor2=scipy.signal.convolve2d(spec_audio, spec_fonema, mode='full', boundary='fill', fillvalue=0)
plt.imshow(cor2)
plt.show()

'''

#Esta función convoluciona una matriz de una imagen con un filtro
def conv3d(kernel_diag1, data1):
    #se determinan las medidas de la matriz
    t_ancho1=len(data1)
    t_alto1=len(data1[0])
    #se define las distancias del centro del kernel a los extremos
    ancho_kernel=math.floor(len(kernel_diag1)/2)
    largo_kernel=math.floor(len(kernel_diag1[0])/2)
    #se define cuál de los dos lados es más grande
    if len(kernel_diag1)>=len(kernel_diag1[0]):
        lado=len(kernel_diag1)
    else:
        lado=len(kernel_diag1[0])
    #se crea una matriz cuadrada de ceros del tamaño del mayor lado 
    #del kernel
    kernel_diag=np.zeros((lado,lado))
    medio_k=math.floor(len(kernel_diag)/2)
    #Se creo una matriz más grande que la de la imagen rellenando los bordes
    #con ceros para no haber problemas al convolucionar la matriz con el 
    #kernel
    data=np.zeros(  (t_ancho1 + 2 *medio_k ,t_alto1 +2*medio_k   )     )
    #se determinan las medidas de la nueva matriz de la imagen   
    t_ancho=len(data)
    t_alto=len(data[0])
    lim_diag= medio_k - 1
    #se crea una matriz de caracteristicas y una de caracteristicas
    #rectificadas con dimensiones las guardadas de la matriz de la imagen
    #antes de ser rellenada con ceros
    features_map=  np.zeros( (t_ancho1,t_alto1))
    #se llena con ceros y los datos del kernel original la nueva matriz de kernel
    
    for kl1 in range(medio_k - ancho_kernel,  lado - medio_k + ancho_kernel):
        for kl2 in range(medio_k - largo_kernel, lado - medio_k + largo_kernel):
            kernel_diag[kl1][kl2]=kernel_diag1[kl1- medio_k + ancho_kernel][kl2 - medio_k + largo_kernel]
    print("MAPA DE FONEMA ARREGLADA, TAMAÑO")
    print(kernel_diag.shape)
    plt.imshow(kernel_diag)
    plt.show()
    
    #se rellena con ceros los bordes de la nueva matriz de la imagen
    for m1 in range(t_ancho):
        for m2 in range(t_alto):
                if (m1 <=lim_diag) or (m1>=t_ancho-lim_diag -1 ) or (m2<=lim_diag ) or (m2>=t_alto-lim_diag -1):
                    data[m1][m2]=0
                else:
                    data[m1][m2]=data1[m1-medio_k][m2-medio_k]
    print("MAPA CON CEROS, TAMAÑO")
    print(data.shape)
    plt.imshow(data)
    plt.show()
    bias=random.randint(1,5)

    #se convoluciona el kernel con la nueva matriz de la imagen
    for i1 in range(medio_k,t_ancho-medio_k):
        for i2 in range(medio_k,t_alto-medio_k):
            features_map[i1-medio_k][i2-medio_k]=0
            for k1 in range(len(kernel_diag)):
                for k2 in range(len(kernel_diag)):
                    features_map[i1-medio_k][i2-medio_k]= features_map[i1-medio_k][i2-medio_k] + data[i1+k1-medio_k][i2 + k2-medio_k] * kernel_diag[k1][k2]
            features_map[i1-medio_k][i2-medio_k]= features_map[i1-medio_k][i2-medio_k]  + bias
    print("MAPA DE CARACTERISTICAS, TAMAÑO")
    print(features_map.shape)
    #se muestra el resultado de la convolución
    plt.imshow(features_map)
    plt.show()
    
conv3d(spec_fonema, spec_audio)

