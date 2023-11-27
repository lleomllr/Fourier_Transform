# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 18:47:01 2023

@author: meill
"""

import TransformeeDeFourier.DFT1D as dft1d
import TransformeeDeFourier.DFTD2D as dft2d
import TransformeeDeFourier.FFT as fft
import TransformeeDeFourier.FFT2D as FFT2D

import numpy as np
import cmath
import time
import matplotlib.pyplot as plt
from PIL import Image

#util
def ajuster_taille_image(image_path):
    # Ouvrir l'image avec Pillow
    image = Image.open(image_path)

    # Obtenir les dimensions actuelles de l'image
    largeur, hauteur = image.size

    # Trouver la puissance de 2 supérieure ou égale aux dimensions de l'image
    puissance_deux_largeur = 1
    while puissance_deux_largeur < largeur:
        puissance_deux_largeur *= 2

    puissance_deux_hauteur = 1
    while puissance_deux_hauteur < hauteur:
        puissance_deux_hauteur *= 2

    # Si la taille actuelle est déjà une puissance de 2, ne rien faire
    if puissance_deux_largeur == largeur and puissance_deux_hauteur == hauteur:
        return image

    # Ajuster la taille de l'image à la puissance de 2 la plus proche
    nouvelle_taille = (puissance_deux_largeur, puissance_deux_hauteur)
    image_redimensionnee = image.resize(nouvelle_taille)

    return image_redimensionnee


#_______________________________________________________________________
#Test de la Transformée discrète 1D directe et inverse
adresse_image = 'TransformeeDeFourier/calimero.jpg'
image = Image.open(adresse_image).convert('L')
image_data = np.array(image)

debut_time_dft1d = time.time()
#resultat_dft = dft1d.direct(image_data[0])
resultat_dft = np.array([dft1d.direct(ligne) for ligne in image_data])
fin_time_dft1d = time.time()

debut_time_idft1d = time.time()
#resultat_idft = dft1d.inverse(resultat_dft)
resultat_idft = np.array([dft1d.inverse(ligne) for ligne in resultat_dft])
fin_time_idft1d = time.time()

temps_dft1d = fin_time_dft1d - debut_time_dft1d
temps_idft1d = fin_time_idft1d - debut_time_idft1d

plt.figure(figsize=(6, 6))
plt.imshow(image_data, cmap='gray')
plt.title('Image Originale')
plt.show()

magnitude_spectrum = [np.abs(freq) for freq in resultat_dft]
plt.figure(figsize=(6, 6))
plt.plot(magnitude_spectrum)
plt.title('Magnitude Spectrum of DFT Result')
plt.show()

#real_idft_resultat = [np.real(val) for val in resultat_idft]
plt.figure(figsize=(6, 6))
#plt.plot(real_idft_resultat)
plt.imshow(np.real(resultat_idft), cmap='gray')
plt.title('Résultat de la Transformée Inverse')
plt.show()

print(f"Temps estimé pour la DFT directe : {temps_dft1d:.4f} seconds")
print(f"Temps estimé pour la DFT Inverse : {temps_idft1d:.4f} seconds")

"""

#_______________________________________________________________________
#Test de la Transformée de Fourier discrète 1D Rapide directe et inverse 
chemin = 'TransformeeDeFourier/calimero.jpg'

data_ajustee = ajuster_taille_image(chemin)
image_gris = data_ajustee.convert('L')
data = np.array(image_gris)

debut = time.time()
resultat_fft = [fft.direct(row.tolist()) for row in data]
fin = time.time()

start_time_ifft = time.time()
ifft_resultat = [fft.inverse(row) for row in resultat_fft]
end_time_ifft = time.time()

temps_estime = fin - debut
temps_ifft = end_time_ifft - start_time_ifft


plt.figure(figsize=(12,6))
plt.subplot(1, 3, 1)
plt.title("Image Originale")
plt.imshow(data, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Après FFT")
plt.imshow(np.log(np.abs(resultat_fft) + 1), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Après IFFT : Reconstruction")
plt.imshow(np.real(ifft_resultat), cmap='gray')

plt.show()

print(f"Temps estimé pour la FFT : {temps_estime:.4f} seconds")
print(f"Temps estimé pour la IFFT : {temps_ifft:.4f} seconds")



#_______________________________________________________________________
#Test de la Transformée de Fourier discrète 2D directe et inverse 
matrice_test = np.random.rand(4, 4)
matrice_tfd2d_directe = dft2d.direct(matrice_test)

matrice_tfd2d_inverse = dft2d.inverse(matrice_tfd2d_directe)

print("Matrice originale:\n", matrice_test)
print("Matrice après TFD2D inverse:\n", matrice_tfd2d_inverse)

erreur = np.linalg.norm(matrice_test - matrice_tfd2d_inverse)
print("Erreur entre la matrice originale et la récupérée par TFD2D inverse:", erreur)


chemin = 'TransformeeDeFourier/lena.jpg'
image = Image.open(chemin)
image_gris = image.convert('L')
data = np.array(image_gris)

ajuster_taille_image(chemin)


debut = time.time()
resultat_dft2d = dft2d.direct(data)
fin = time.time()

start_time_ifft = time.time()
resultat_idft2d = dft2d.inverse(resultat_dft2d)
end_time_ifft = time.time()

temps_estime = fin - debut
temps_ifft = end_time_ifft - start_time_ifft


plt.figure(figsize=(12,6))
plt.subplot(1, 3, 1)
plt.title("Image Originale")
plt.imshow(data, cmap='gray')


plt.subplot(1, 3, 2)
plt.title("Après DFT2D")
plt.imshow(np.log(np.abs(resultat_dft2d) + 1), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Après IDFT2d : Reconstruction")
plt.imshow(np.real(resultat_idft2d), cmap='gray')

plt.show()

print(f"Temps estimé pour la DFT2D : {temps_estime:.4f} seconds")
print(f"Temps estimé pour la IDFT2D : {temps_ifft:.4f} seconds")


#_______________________________________________________________________
#Test de la Transformée de Fourier discrète 2D rapide directe et inverse
chemin = '/Users/meill/OneDrive/Bureau/L3/Maths/Projet/TransformeeDeFourier/lena.jpg'
image = Image.open(chemin)
image_gris = image.convert('L')
data = np.array(image_gris)

ajuster_taille_image(chemin)


debut = time.time()
resultat_fft2d = fft2d.direct(data)
fin = time.time()

start_time_ifft = time.time()
resultat_ifft2d = fft2d.inverse(resultat_fft2d)
end_time_ifft = time.time()

temps_estime = fin - debut
temps_ifft = end_time_ifft - start_time_ifft


debut_np = time.time()
resultat_fft2d_np = np.fft.fft2(data)
fin_np = time.time()


start_time_ifft_np = time.time()
ifft2d_np_resultat = np.fft.ifft2(resultat_fft2d_np)
end_time_ifft_np = time.time()


tmps_estime = fin_np - debut_np
temps_ifft_np = end_time_ifft_np - start_time_ifft_np


plt.figure(figsize=(12,6))
plt.subplot(1, 3, 1)
plt.title("Image Originale")
plt.imshow(data, cmap='gray')


plt.subplot(1, 3, 2)
plt.title("Après FFT2D")
plt.imshow(np.log(np.abs(resultat_fft2d) + 1), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Après IFFT2d : Reconstruction")
plt.imshow(np.real(resultat_ifft2d), cmap='gray')

plt.show()

print(f"Temps estimé pour la FFT2D : {temps_estime:.4f} seconds")
print(f"Temps estimé pour la IFFT2D : {temps_ifft:.4f} seconds")


plt.subplot(1, 3, 2)
plt.title("Après FFT2D")
plt.imshow(np.log(np.abs(resultat_fft2d_np) + 1), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Après IFFT2d : Reconstruction")
plt.imshow(np.real(ifft2d_np_resultat), cmap='gray')

plt.show()

print(f"Temps estimé pour la np FFT2D : {tmps_estime:.4f} seconds")
print(f"Temps estimé pour la np IFFT2D : {temps_ifft_np:.4f} seconds")
"""
