# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 17:05:36 2023

@author: meill
"""

"""Fast Fourier Transform 2D """

"""Transformée de Fourier rapide 2D"""

import cmath
import TransformeeDeFourier.FFT as fft

#%%Passage de la Transformée de Fourier Discrète 2D de O(N^4) à O(2 * N^2.log2(N))

def transpose(tab):
    #initialisation du nombre de lignes et de colonnes de la matrice
    I = len(tab)
    J = len(tab[0])
    
    #création d'une nouvelle matrice avec J lignes et I colonnes, i.e l'inverse de la matrice tab
    new_tab = [[0 for i in range(I)] for j in range(J)]
    #parcours de la matrice et transposition de la matrice
    for i in range(I):
        for j in range(J):
            new_tab[j][i] = tab[i][j]
    return new_tab

def direct(tab):
    #initialisation du nombre de lignes et de colonnes de la matrice
    I = len(tab)
    J = len(tab[0])
    #fft appliquée sur toutes les lignes de la matrice
    for i in range(I):
        tab[i] = fft.direct(tab[i])
    #matrice transposée => les lignes deviennent des colonnes et inversement   
    tab = transpose(tab)
    #fft appliquée aux nouvelles lignes de la matrices (anciennes colonnes avant transposée précédente)
    for j in range(J):
        tab[i]=fft.direct(tab[i])
    #on transpose à nouveau la matrice pour revenir à sa structure originale
    tab = transpose(tab)
    #la matrice transformée est renvoyée
    return tab

def inverse(tab):
    #initialisation du nombre de lignes et de colonnes de la matrice
    I = len(tab)
    J = len(tab[0])
    #fft inverse appliquée sur toutes les lignes de la matrice
    for i in range(I):
        tab[i] = fft.inverse(tab[i])
    tab = transpose(tab)
    for i in range(J):
        tab[i] = fft.inverse(tab[i])
    tab = transpose(tab)
    return tab
    