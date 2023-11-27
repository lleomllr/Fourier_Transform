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
    #traite chaque colonne de la matrice
    colonnes = [fft.direct([ligne[i] for ligne in matrix]) for i in range(len(matrix[0]))]
    #pour chaque colonne, une liste est construite en prenant le i-ème elem de chaque ligne. 
    #Puis application de fft directe a cette liste (donc sur chaque colonne)
    
    #utilisation de zip pour transposer la liste des colonnes transformées puis map et list pour convertir le resultat en une liste de listes
    transposee = list(map(list, zip(*colonnes)))
    #applique fft direct sur chaque ligne de la mtrice transposée
    #effectue ainsi fft sur les lignes de la matrice originale (qui sont maintenant des colonnes de la matrice transposée)
    lignes = [fft.direct(ligne) for ligne in transposee]
    
    return lignes

def inverse(matrix):
    #application de fft inverse sur chaque ligne de la matrice
    lignes = [fft.inverse(ligne) for ligne in matrix]
    #transpose la matrice résultante en utilisant zip 
    transposee = list(map(list, zip(*lignes)))
    #fft inverse sur chaque ligne de la matrice transposée (anciennes colonnes de la matrice originale)
    colonnes = [fft.inverse(ligne) for ligne in transposee]
    #transpose de nouveau pour remettre les lignes et les colonnes dans leur ordre original
    resultat = list(map(list, zip(*colonnes)))
    return resultat

#util
def ajuster_taille_matrice(matrice):
    # Vérifier la taille des lignes et colonnes
    lignes_actuelles = len(matrice)
    colonnes_actuelles = len(matrice[0]) if lignes_actuelles > 0 else 0

    # Trouver la puissance de 2 supérieure ou égale au nombre de lignes et de colonnes
    puissance_deux_lignes = 1
    while puissance_deux_lignes < lignes_actuelles:
        puissance_deux_lignes *= 2

    puissance_deux_colonnes = 1
    while puissance_deux_colonnes < colonnes_actuelles:
        puissance_deux_colonnes *= 2

    # Si la taille actuelle est déjà une puissance de 2, ne rien faire
    if puissance_deux_lignes == lignes_actuelles and puissance_deux_colonnes == colonnes_actuelles:
        return matrice

    # Ajouter des lignes ou des colonnes de zéros au besoin
    nouvelle_taille_lignes = puissance_deux_lignes
    nouvelle_taille_colonnes = puissance_deux_colonnes

    for i in range(lignes_actuelles, nouvelle_taille_lignes):
        matrice.append([0] * colonnes_actuelles)

    for row in matrice:
        row.extend([0] * (nouvelle_taille_colonnes - colonnes_actuelles))

    return matrice


#Test avec matrice et comparaison avec la fft2d de la librairie numpy
I = [[1, 2, 3, 4,5,6,7], [1, 2, 3, 4, 5, 6, 7]]
X = ajuster_taille_matrice(I)
F = direct(X)
print(F)
print("\n")
print(inverse(F))
print("\n")

J = np.fft.fft2(I)
print(J)
print("\n\n\n\n\n")
print(np.fft.ifft2(J))

