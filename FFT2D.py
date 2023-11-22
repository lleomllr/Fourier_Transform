# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 17:05:36 2023

@author: meill
"""

"""Fast Fourier Transform 2D """

"""Transformée de Fourier rapide 2D"""

import TransformeeDeFourier.FFT as fft
import numpy as np

#%%Passage de la Transformée de Fourier Discrète 2D de O(N^4) à O(2 * N^2.log2(N))
def direct(matrix):
    colonnes = [fft.direct([ligne[i] for ligne in matrix]) for i in range(len(matrix[0]))]

    transposee = list(map(list, zip(*colonnes)))
    
    lignes = [fft.direct(ligne) for ligne in transposee]
    
    return lignes

def inverse(matrix):
    
    lignes = [fft.inverse(ligne) for ligne in matrix]

    transposee = list(map(list, zip(*lignes)))
    
    colonnes = [fft.inverse(ligne) for ligne in transposee]
    
    resultat = list(map(list, zip(*colonnes)))
    return resultat

#Test avec matrice et comparaison avec la fft2d de la librairie numpy
I = [[1, 2, 3, 4,5,6,7,8], [1, 2, 3, 4, 5, 6, 7, 8]]
F = direct(I)
print(F)
print("\n")
print(inverse(F))
print("\n")

J = np.fft.fft2(I)
print(J)
print("\n\n\n\n\n")
print(np.fft.ifft2(J))

"""
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
  """  
