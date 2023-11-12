# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:06:15 2023

@author: meill
"""

"""Transformée de Fourier discrète bidimensionnelle"""

"""DISCRETE FOURIER TRANSFORM 2D"""


import cmath
import numpy as np
import matplotlib.image as img
import TransformeeDeFourier.DFT1D as dft1d


#%%Transformée de Fourier discrète 2D directe et inverse
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
    
    #parcours de chaque ligne de la matrice et application de la dft1d
    for i in range(I):
        tab[i] = dft1d.direct(tab[i])
    tab = transpose(tab)
    #parcours de chaque colonne (mtn lignes après transposition) de la matrice et application de la dft1d
    for i in range(J):
        tab[i] = dft1d.direct(tab[i])
    tab = transpose(tab)
    return tab

def inverse(tab):
    #initialisation du nombre de lignes et de colonnes de la matrice
    I = len(tab)
    J = len(tab[0])
    #parcours de chaque ligne de la matrice et application de l'inverse de la dft1d
    for i in range(I):
        tab[i] = dft1d.inverse(tab[i])
    tab = transpose(tab)
    #parcours de chaque colonne (mtn lignes après transposition) de la matrice et application de l'inverse de la dft1d
    for i in range(J):
        tab[i] = dft1d.inverse(tab[i])
    tab = transpose(tab)
    return tab
"""

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
