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
def direct(tab):
    #nb de ligne du tab
    N = len(tab)
    #nb de colonne du tab
    M = len(tab[0]) 
    
    #création d'un tab de même taille que tab
    new_tab = [[0 for j in range(M)] for i in range(N)]
    
    #parcourt de chaque ligne
    for i in range(N):
        #application de la dft1D direct sur la i_ème ligne de tab
        new_tab[i] = dft1d.direct(tab[i])
        
    #parcourt chaque colonne
    for j in range(M):
        #extrait la j-ième colonne de new_tab dans une liste colonne
        colonne = [new_tab[i][j] for i in range(N)]
        #application de la dft1D direct sur la colonne extraite
        transformee_col = dft1d.direct(colonne)
        #parcourt des éléments de la colonne transformee 
        for i in  range(N):
            #maj pour le j-ième elemn de la i-ème ligne de new_tab avec la valeur transformee
            new_tab[i][j] = transformee_col[i]
    
    return new_tab


def inverse(tab):
    N = len(tab)
    M = len(tab[0]) 
    
    new_tab = [[0 for i in range(M)] for j in range(N)]
    
    for i in range(N):
        new_tab[i] = dft1d.inverse(tab[i])
        
    for j in range(M):
        colonne = [new_tab[i][j] for i in range(N)]
        transformee_col = dft1d.inverse(colonne)
        for i in  range(N):
            new_tab[i][j] = transformee_col[i]
    
    return new_tab

#Util
def prochaine_puisde2(n): 
    taille = 1
    while taille < n: 
        taille *=2
    return taille 

def ajust_ligne(tab, taille_cible): 
    tab.extend([0] * (taille_cible - len(tab)))
    
    
def ajust_matrice(matrice): 
    taille_max_ligne = max(len(ligne) for ligne in matrice)
    taille_cible_ligne = prochaine_puisde2(taille_max_ligne)
    for ligne in matrice: 
        ajust_ligne(ligne, taille_cible_ligne)
        
    taille_max_col = len(matrice)
    taille_cible_col = prochaine_puisde2(taille_max_col)
    while len(matrice) < taille_cible_col: 
        matrice.append([0] * taille_cible_ligne)

#Test 
matrice_test = [[1, 2, 3], [1, 2, 3]]
ajust_matrice(matrice_test)
matrice_tfd2d_directe = direct(matrice_test)

matrice_tfd2d_inverse = inverse(matrice_tfd2d_directe)

print("Matrice originale:\n", matrice_test)
print("\n")
print("Matrice après TFD2D inverse:\n", matrice_tfd2d_inverse)
print("\n")

erreur = np.linalg.norm(matrice_test - matrice_tfd2d_inverse)
print("Erreur entre la matrice originale et la récupérée par TFD2D inverse:", erreur)

erreur = np.linalg.norm(matrice_test - matrice_tfd2d_inverse)
print("Erreur entre la matrice originale et la récupérée par TFD2D inverse:", erreur)
