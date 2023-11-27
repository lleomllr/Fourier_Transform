# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 17:34:30 2023

@author: meill
"""

"""Transformée de Fourier discrète unidimensionnelle"""

"""DISCRETE FOURIER TRANSFORM 1D"""

import cmath

#%%Transformee de Fourier Discrète 1D directe et inverse

def direct(tab):
    #longueur du tableau pris en entrée
    N = len(tab)
    #initialisation avec des zéros d'un nouveau tableau de même taille
    new_tab = [0]*N
    #parcours de la matrice
    for m in range(0, N):
        for n in range(0, N):
            new_tab[m] = new_tab[m] + tab[n] * cmath.exp(-2 * 1j * cmath.pi * m * n / N);
    return new_tab

def inverse(tab):
    #longueur du tableau pris en entrée
    N = len(tab)
    #initialisation avec des zéros d'un nouveau tableau de même taille
    new_tab = [0]*N
    #parcours de la matrice
    for m in range(0, N):
        for n in range(0, N):
            #pour chaque m, calcule somme des n
            #chaque terme de la somme est le produit d'un n terme et de la valeur exponentielle de la formule
            #Lorsqu'on prend la Transformée de Fourier Inverse, nous devons obtenir le signal original et non pas une version atténuée ou amplifiée,
            #c'est pourquoi nous divisons le tout par 1/N
            new_tab[m] = new_tab[m] + (tab[n] * cmath.exp(2 * 1j * cmath.pi * m * n / N))/N;
    return new_tab

def verifPuissance(tab): 
    taille_actu = len(tab)
    
    taille_demand = 1
    while taille_demand < taille_actu: 
        taille_demand *=2
        
    tab.extend([0] * (taille_demand - taille_actu))

I=[1,2,3,4,5,6,7]
verifPuissance(I)
F = direct(I)
print(F)
print("\n")
print(inverse(F))
print(F)
print("\n")
print(inverse(F))
