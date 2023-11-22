# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 18:47:01 2023

@author: meill
"""

import TransformeeDeFourier.DFT1D as dft1d
import TransformeeDeFourier.DFTD2D as dft2d
import TransformeeDeFourier.FFT as fft
import TransformeeDeFourier.FFT2D as FFT2D
import TransformeeDeFourier.util as utl

import numpy as np
import cmath
import time
import matplotlib.pyplot as plt
from PIL import Image

#Test de la Transformée discrète 1D directe et inverse
adresse_image = '/Users/meill/OneDrive/Bureau/L3/Maths/Projet/TransformeeDeFourier/calimero.jpg'
image = Image.open(adresse_image).convert('L')
image_data = np.array(image)

plt.figure(figsize=(6, 6))
plt.imshow(image_data, cmap='gray')
plt.title('Image Originale')
plt.show()

debut_time_dft1d = time.time()
resultat_dft = dft1d.direct(image_data[0])
fin_time_dft1d = time.time()

debut_time_idft1d = time.time()
resultat_idft = dft1d.inverse(resultat_dft)
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

real_idft_resultat = [np.real(val) for val in resultat_idft]
plt.figure(figsize=(6, 6))
plt.plot(real_idft_resultat)
plt.title('Résultat de la Transformée Inverse')
plt.show()

print(f"Temps estimé pour la DFT directe : {temps_dft1d:.4f} seconds")
print(f"Temps estimé pour la DFT Inverse : {temps_idft1d:.4f} seconds")

----------------------------------------------------------------------------------------------

#Test de la Transformée de Fourier discrète 1D Rapide directe et inverse 
chemin = '/Users/meill/OneDrive/Bureau/L3/Maths/Projet/TransformeeDeFourier/calimero.jpg'
image = Image.open(chemin)
image_gris = image.convert('L')
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
print(f"Temps estimé pour la IFFT : {temps_estime:.4f} seconds")


------------------------------------------------------------------------------------------------------------------
#Test de la Transformée de Fourier discrète 2D directe et inverse 
matrice_test = np.random.rand(4, 4)
matrice_tfd2d_directe = dft2d.direct(matrice_test)

matrice_tfd2d_inverse = dft2d.inverse(matrice_tfd2d_directe)

print("Matrice originale:\n", matrice_test)
print("Matrice après TFD2D inverse:\n", matrice_tfd2d_inverse)

erreur = np.linalg.norm(matrice_test - matrice_tfd2d_inverse)
print("Erreur entre la matrice originale et la récupérée par TFD2D inverse:", erreur)


