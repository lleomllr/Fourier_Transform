# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 17:31:40 2023

@author: meill
"""

"""Fast Fourier Transform"""

"""Transformée de Fourier rapide"""

import cmath

#%%Passage de la Transformée de Fourier Discrète 1D de O(N^2) à O(Nlog2(N))

def direct(tab):
    N = len(tab)
    if N <=1:
        return tab
    paires = direct(tab[0::2])
    impaires = direct(tab[1::2])
    
    new_tab = [cmath.exp((-2j * cmath.pi * k)/N) * impaires[k%(N//2)] for k in range(N)]
    
    return [paires[k%(N//2)]+new_tab[k] for k in range(N)]

def inverse(tab):
    N = len(tab)
    if N <=1:
        return tab
    #si non , appel initial et stock la taille du tableau
    #permet de distinguer l'appel initial des appels récursifs
    if not hasattr(inverse, "longueur1erTab"):
        inverse.longueur1erTab = N
    paires = inverse(tab[0::2])
    impaires = inverse(tab[1::2])
    
    new_tab = [cmath.exp((-2j * cmath.pi * k)/N) * impaires[k%(N//2)] for k in range(N)]
 
    #vérifie si la taille du tableau est égale à la taille du tableau lors de l'appel initial de la fonction inverse
    if len(tab) == inverse.longueur1erTab:
        #suppression de l'attribut "" qui servait à mémoriser la taille du tableau d'entrée
        delattr(inverse, "longueur1erTab")
        #combination des paires et du nouveau tableau
        #résultat normalisé en divisant chaque élément par la taille du tableau
        return [((1 / len(tab)) * (paires[k%(N//2)] + new_tab[k])) for k in range(N)]
    #si nous ne sommes pas dans un appel initial mais récursif, les élements sont ajoutés sans normalisation
    else:
        return [(paires[k%(N//2)] + new_tab[k]) for k in range(N)]


#nouvelle version fft direct 1D : 
def direct(tab, w): 
    N = len(tab)
    if N <= 1:  #si N constant alors on retourne simplement le tableau  
        return tab
    else: 
        paires = (tab[0:2])   #tableau pair partant de 0 (deux en deux)   
        impaires = (tab[1:2]) #tableau impair partant de 1 (deux en deux) 

        direct(paires)        #appels recursifs sur les indices des tabs pairs et impairs
        direct(impaires)

        for k in range(0, N//2): #parcours des indices des indices venant de l'appel recursif
            new_tab =[cmath.exp((-2j * cmath.pi * k / N)) * impaires[k]] #application de la formule et multiplié par les impairs 
            tab[k] = paires[k] + new_tab  
            tab[N//2 + k] = paires[k] - new_tab
