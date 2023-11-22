# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:06:15 2023

@author: meill
"""

"""Transformée de Fourier discrète bidimensionnelle"""

"""DISCRETE FOURIER TRANSFORM 2D"""

import TransformeeDeFourier.DFT1D as dft1d
import numpy as np


#%%Transformée de Fourier discrète 2D directe et inverse
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
    N = len(tab)
    M = len(tab[0]) if N > 0 else 0
    
    new_tab = [[0 for j in range(M)] for i in range(N)]
    
    for i in range(N):
        new_tab[i] = dft1d.direct(tab[i])
        
    for j in range(M):
        colonne = [new_tab[i][j] for i in range(N)]
        transformee_col = dft1d.direct(colonne)
        for i in  range(N):
            new_tab[i][j] = transformee_col[i]
    
    return new_tab


def inverse(tab):
    N = len(tab)
    M = len(tab[0]) if N > 0 else 0
    
    new_tab = [[0 for i in range(M)] for j in range(N)]
    
    for i in range(N):
        new_tab[i] = dft1d.inverse(tab[i])
        
    for j in range(M):
        colonne = [new_tab[i][j] for i in range(N)]
        transformee_col = dft1d.inverse(colonne)
        for i in  range(N):
            new_tab[i][j] = transformee_col[i]
    
    return new_tab

#Test 
matrice_test = np.random.rand(4, 4)
matrice_tfd2d_directe = direct(matrice_test)

matrice_tfd2d_inverse = inverse(matrice_tfd2d_directe)

print("Matrice originale:\n", matrice_test)
print("Matrice après TFD2D inverse:\n", matrice_tfd2d_inverse)

erreur = np.linalg.norm(matrice_test - matrice_tfd2d_inverse)
print("Erreur entre la matrice originale et la récupérée par TFD2D inverse:", erreur)
