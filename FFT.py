# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 17:31:40 2023

@author: meill
"""

"""Fast Fourier Transform"""

"""Transformée de Fourier rapide"""

#%%Passage de la Transformée de Fourier Discrète 1D de O(N^2) à O(Nlog2(N))

import cmath
import numpy as np

#implémentation récursive
def direct(tab):
    N = len(tab)
    if N <= 1: 
        return tab
    
    new_tab = [0]*N
    
    paires = direct(tab[0::2])
    impaires =  direct(tab[1::2])
    
    #racine de l'unité
    w = cmath.exp((-2j*cmath.pi)/N)
    
    for u in range(N//2):
        #somme de l'élem correspondant dans le tab paire et le produit 'w' élévé à la puissance k et l'elem correspondant dans le tab des impaires
        new_tab[u] = paires[u] + w**u*impaires[u]
        #différence des mêmes termes, ce qui reflète la symétrie de la FFT
        new_tab[u+N//2] = paires[u] - w**u*impaires[u]
    return new_tab

def inverse(tab):
    N = len(tab)
    if N <= 1: 
        return tab
    
    new_tab = [0]*N
    
    paires = inverse(tab[0::2])
    impaires =  inverse(tab[1::2])
    
    #racine de l'unité
    w = cmath.exp((2j*cmath.pi)/N)
    
    for u in range(N//2):
        #somme de l'élem correspondant dans le tab paire et le produit 'w' élévé à la puissance k et l'elem correspondant dans le tab des impaires
        new_tab[u] = (paires[u] + (w**u)*impaires[u])/2
        #différence des mêmes termes, ce qui reflète la symétrie de la FFT
        new_tab[u+N//2] = (paires[u] - w**u*impaires[u])/2
    return new_tab


def verifPuissance(tab): 
    taille_actu = len(tab)
    
    taille_demand = 1
    while taille_demand < taille_actu: 
        taille_demand *=2
        
    tab.extend([0] * (taille_demand - taille_actu))


#Test avec matrice et comparaison avec la fft de la librairie numpy
I = [1, 2, 3, 4,5,6,7]
verifPuissance(I)
F = direct(I)
print(F)
print("\n")
print(inverse(F))
print("\n")

J = np.fft.fft(I)
print(J)
print("\n\n\n\n\n")
print(np.fft.ifft(J))


#%%implémentation itérative basée sur l'algo de Cooley-Tuckey

def iterative_direct(tab):
    #taille de l'entrée 
    N = len(tab)
    #liste d'indices des élements dans tab
    indices = list(range(N))
    #stock l'indice actuel dans l'ordre bit-reversal
    rev = 0
    #boucle qui itère sur chaque element de la liste à l'exception du premier (qui est toujours 0)
    for i in range(1, N):
        bit = N >> 1
        while rev >= bit:
            rev -= bit
            bit >>= 1
        rev += bit
        if i < rev:
            indices[i], indices[rev] = indices[rev], indices[i]

    # Cooley-Tukey
    tab = [tab[i] for i in indices]
    #variable représentant la taille des DFT à fusionner
    #commence à 2 et double à chaque itération jusqu'a atteindre N
    m = 2
    while m <= N:
        #moitié de m utilisée pour diviser les DFT en parties paires et impaires
        half_m = m // 2
    
        w_m = cmath.exp(-2j * cmath.pi / m)
        
        for k in range(0, N, m):
            w = 1
            for j in range(half_m):
                t = w * tab[k + j + half_m]
                u = tab[k + j]
                tab[k + j] = u + t
                tab[k + j + half_m] = u - t
                w *= w_m
        m *= 2
    return tab
    

def ifft_iterative(tab):
    N = len(tab)
    #créa d'une nouvelle ligne qui contient les conjugués de tous les élements de tab 
    #La conjugaison d'un nb complexe => changement de signe de sa partie imaginaire
    tab_conj = [val.conjugate() for val in tab]
    
    #application de FFT à la liste des conjugués complexes => nous rapproche du domaine temporel en venant du domaine fréquentiel
    new_tab = iterative_direct(tab_conj)
    
    #conjugué de la fft conjuguée
    #rétablit les signes des parties imaginaires des nb complexes
    new_tab = [val.conjugate() for val in new_tab]
    
    #chaque elem de la liste résultante est divisé par le nb total d'elemn de N 
    #nécessaire car la fft calcule la somme des produits et non la moyenne
    #division par N normalise le résultat pour obtenir la moyenne
    return [val / N for val in new_tab]
