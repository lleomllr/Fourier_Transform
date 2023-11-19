# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:04:38 2023

@author: meill
"""
"""FONCTIONS UTILES AUX TESTS DES FONCTIONS"""


from PIL import Image 
import numpy as np


#charge une image et retourne le tableau correspondant
def chargerImage(chemin):
    image = Image.open(chemin) # Ouvre une image 
    return np.asarray(image)   # Converti l'image en un tableau


#affiche l'image repreésentée par le tableau
def showImg(tab_data):
    img = Image.fromarray(tab_data) # Converti le tableau en une image PIL
    img.show()
    

#retourne un couple contenant les dimensions du tableau
def dimension_tab(tab):
    # tuple contenant les dimensions du tableau
    (haut, larg, *autre) = tab.shape # *autre capture les autres éléments du tuple qui ne sont pas explicitement assignés.
    return (haut, larg)


#converti un tableau rgb en un tableau de niveaux de gris
def rgb_tab_to_gray(tab):
    (haut, larg) = dimension_tab(tab)
    #tableau vide de même hauteur et largeur que l'image originale
    valeurs_gris = np.empty([haut, larg], dtype=np.complex128)
    
    #parcourt de chaque pixel de l'image
    for i in range(haut):
        for j in range(larg):
            #Calcul de valeur de gris pour chaque pixel
            valeurs_gris[i, j] = (int(tab[i, j][0]) + int(tab[i, j][1]) + int(tab[i, j][2])) / 3
    return valeurs_gris


#converti un tableau avec des pixels de 0 à 255 vers des pixels allant de 0 à 1 (float)
def normalize_img(tab):
    (haut, larg) = dimension_tab(tab)
    
    valeurs = np.empty([haut, larg], dtype=np.complex128)
    
    #parcout des pixels de l'image
    for i in range(haut):
        for j in range(larg):
            #divise la val de chaque pixel par 255 et converti les val de pixel de 0-255 à 0-1
            valeurs[i, j] = tab[i, j] / 255.0
    return valeurs


#converti un tableau de pixels avec des valeurs entre 0 et 1 vers des valeurs entre 0 et 255
def unnormalize_img(tab):
    (haut, larg) = dimension_tab(tab)
    
    valeurs = np.empty([haut, larg])
    
    for i in range(haut):
        for j in range(larg):
            #multiplie la val de chaque pixel par 255 et converti les val de pixel de 0-1 à 0-255
            valeurs[i, j] = tab[i, j] * 255.0
    return valeurs