# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 18:47:01 2023

@author: meill
"""

import TransformeeDeFourier.DFT1D as dft1d
import TransformeeDeFourier.DFTD2D as dft2d
import TransformeeDeFourier.FFT as fft
import TransformeeDeFourier.FFT2D as FFT2D
import copy
import matplotlib.image as img
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy

if __name__ == '__main__':
    #transformée 1D direct
    tab1DD = [12, 45,20,30,42,123,3841,123,43,219,4312,1232,76,24,87,996]  # exemple de tableau 1d
    tab2DD = [[13, 5,980,20,3,23,341,63,78,29,432,932,778,36,70,1000],
    [1, 5,2,3,42,23,31,13,43,19,412,122,7,4,8744,94296]]# exemple de tableau 2d
    tab1DF = tab1DD.copy()
    tab2DF = tab2DD.copy()
    
    tabFTD = dft1d.inverse(dft1d.direct(tab1DD))
    tabFFT = fft.inverse(fft.direct(tab1DF))
    
    tabFTD2 = dft2d.inverse(dft2d.direct(tab2DD))
    tabFFT2 = FFT2D.inverse(FFT2D.direct(tab2DF))
    
    for i in range(len(tabFTD)):
        tabFTD[i]=round(tabFTD[i].real,2)+round(tabFTD[i].imag,2)*1j
    
    for i in range(len(tabFFT)):
        tabFFT[i]=round(tabFFT[i].real,2)+round(tabFFT[i].imag,2)*1j
    
    for i in range(len(tabFTD2)):
        for j in range(len(tabFTD2[0])):
            tabFTD2[i][j]=round(tabFTD2[i][j].real,2)+round(tabFTD2[i][j].imag,2)*1j

    for i in range(len(tabFFT2)):
        for j in range(len(tabFFT2[0])):
            tabFFT2[i][j]=round(tabFFT2[i][j].real,2)+round(tabFFT2[i][j].imag,2)*1j
    
    print("Transformée de Fourier 1D")
    print(tabFTD)
    print("Transformée de Fourier 1D rapide")
    print(tabFFT)
    print("Transformée de Fourier 2D")
    print(tabFTD2)
    print("Transformée de Fourier 2D rapide")
    print(tabFFT2)   
    
    img = cv2.imread('calimero.jpg',0)
    cv2.imshow('De Base',img)
    imgArr = img.tolist()
    
    imgArr = FFT2D.direct(imgArr)
    imgArrRealFFT2 = copy.deepcopy(imgArr)
    
    for i in range(len(imgArrRealFFT2)):
        for j in range(len(imgArrRealFFT2[0])):
            imgArrRealFFT2[i][j] = round(imgArrRealFFT2[i][j].real)
    
    
    img = numpy.uint8(numpy.array(imgArrRealFFT2))
    cv2.imshow('FFT2D', img)
    
    
    imgArr = FFT2D.inverse(imgArr)
    imgArrRealFFT2I = copy.deepcopy(imgArr)
    
    for i in range(len(imgArrRealFFT2I)):
        for j in range(len(imgArrRealFFT2I[0])):
            imgArrRealFFT2I[i][j] = round(imgArrRealFFT2I[i][j].real)
            
    img = numpy.uint8(numpy.array(imgArrRealFFT2I))
    cv2.imshow('FFT2D Inverse', img)
    
    cv2.waitKey(0)