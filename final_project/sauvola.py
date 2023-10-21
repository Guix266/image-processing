# -*- coding: utf-8 -*-
"""
Created on Sun May 10 18:38:40 2020

@author: Guillaume DESERMEAUX
Mini-projet : Sauvola Binarization
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

import affiche


# =============================================================================
# Calculation of mean and standard deviation
# =============================================================================

def Sauv_mean(image, x,y,r):
    """
    Calculate the mean of the pixel (x, y) according to the Niblack's method.
    ie : the mean of the in the square of peaks (x-r, x-r), (x-r, x+r),
                                                (x+r, x-r), (x+r, x+r)

    Parameters
    ----------
    image : 2D array
        the image processed.
    x : INT
        x-Coordinates of the pixel.
    y : INT
        y-Coordinates of the pixel.
    r : INT
        radius of the square considerated in the Niblack's method.

    Returns
    -------
    Mean : FLOAT
        Mean of the pixel (x, y) according to the Niblack's method

    """
    n, m = np.shape(image)
    number_pix = 0
    Mean = 0
    for i in range(x-r, x+r+1):
        for j in range(y-r, y+r+1):
            if (i>=0 and i<n) and (j>=0 and j<m):
                number_pix += 1
                Mean += image[i, j] 
    return(Mean / number_pix)

def Sauv_standard_deviation(image, x, y, r, Mean):
    """
    Calculate the standard deviation of the pixel (x, y) according to the Niblack's method.
    ie : the standard deviation of the in the square of peaks (x-r, x-r), (x-r, x+r),
                                                (x+r, x-r), (x+r, x+r)

    Parameters
    ----------
    image : 2D array
        the image processed.
    x : INT
        x-Coordinates of the pixel.
    y : INT
        y-Coordinates of the pixel.
    r : INT
        radius of the square considerated in the Niblack's method.
    Mean : FLOAT
        the Mean of the pixel (x, y) according to the Niblack's method.

    Returns
    Std : FLOAT
        Standard deviation of the pixel (x, y) according to the Niblack's method

    """
    n, m = np.shape(image)
    number_pix = 0
    Std = 0
    for i in range(x-r, x+r+1):
        for j in range(y-r, y+r+1):
            if (i>=0 and i<n) and (j>=0 and j<m):
                number_pix += 1
                Std += (image[i, j] - Mean)**2 
    return( (Std / number_pix)**(1/2))

test = np.array([ [1, 0, 3, 0],
                  [0, 0, 0, 0],
                  [7, 0, 9, 0],
                  [0, 0, 0, 0]])

# print(Sauv_standard_deviation(test, 0, 2, 1, Sauv_mean(test, 0, 2, 1)))

# =============================================================================
# Function of interpolation
# =============================================================================

def interpol_simple(n, Threshold):
    """ Interpolation simple pour l'algorithme de Sauvola"""
    ns, ms = np.shape(Threshold)
    # For each based pixel
    for i in range(0, ns, n):
        for j in range(0, ms, n):

            ### Case 1 : if there is one summit
            if (i+n)>=ns and (j+n)>=ms:
                Ta = Threshold[i,j]
                # For each pixel in the area
                for k in range(n):
                    for l in range(n):
                        if (k,l)!=(0,0) and (i+k)<ns and (j+l)<ms:
                            Threshold[i+k,j+l] = Ta
            
            ### Case 2 : if there is 2 horizontal summits
            elif (i+n)>=ns:
                Ta = Threshold[i,j]
                Tb = Threshold[i,j+n]
                # For each pixel in the area
                for k in range(n):
                    for l in range(n):
                        if (k,l)!=(0,0) and (i+k)<ns and (j+l)<ms:
                            Threshold[i+k,j+l] = (Ta + Tb)/2
                            
            ### Case 3 : if there is 2 vertical summits
            elif (j+n)>=ms:
                Ta = Threshold[i,j]
                Tc = Threshold[i+n,j]
                # For each pixel in the area
                for k in range(n):
                    for l in range(n):
                        if (k,l)!=(0,0) and (i+k)<ns and (j+l)<ms:
                            Threshold[i+k,j+l] = (Ta + Tc)/2
            
            ### Case 4 : there are the 4 summits
            else:
                Ta = Threshold[i,j]
                Tb = Threshold[i,j+n]
                Tc = Threshold[i+n,j]
                Td = Threshold[i+n,j+n]
                # For each pixel in the area
                for k in range(n):
                    for l in range(n):
                        if (k,l)!=(0,0) and (i+k)<ns and (j+l)<ms:
                            Threshold[i+k,j+l] = (Ta + Tb + Tc + Td)/4
    return( Threshold )
    
# print(interpol_simple(2, test))

def interpol_bili(n, image, Threshold):
    """ Interpolation bilineaire pour l'algorithme de Sauvola.
        Weigths are added in the calculation of the threshold values"""
    ns, ms = np.shape(Threshold)
    # For each based pixel
    for i in range(0, ns, n):
        for j in range(0, ms, n):

            ### Case 1 : if there is one summit
            if (i+n)>=ns and (j+n)>=ms:
                Ta = Threshold[i,j]
                # For each pixel in the area
                for k in range(n):
                    for l in range(n):
                        if (k,l)!=(0,0) and (i+k)<ns and (j+l)<ms:
                            Threshold[i+k,j+l] = Ta
            
            ### Case 2 : if there is 2 horizontal summits
            elif (i+n)>=ns:
                Ta = Threshold[i,j]
                Tb = Threshold[i,j+n]
                # For each pixel in the area
                for k in range(n):
                    for l in range(n):
                        if (k,l)!=(0,0) and (i+k)<ns and (j+l)<ms:
                            Threshold[i+k,j+l] = ( Ta*(n-l) + Tb*l ) / n
                            
            ### Case 3 : if there is 2 vertical summits
            elif (j+n)>=ms:
                Ta = Threshold[i,j]
                Tc = Threshold[i+n,j]
                # For each pixel in the area
                for k in range(n):
                    for l in range(n):
                        if (k,l)!=(0,0) and (i+k)<ns and (j+l)<ms:
                            Threshold[i+k,j+l] = ( Ta*(n-k) + Tc*k ) / n
            
            ### Case 4 : there are the 4 summits
            else:
                Ta = Threshold[i,j]
                Tb = Threshold[i,j+n]
                Tc = Threshold[i+n,j]
                Td = Threshold[i+n,j+n]
                # For each pixel in the area
                for k in range(n):
                    for l in range(n):
                        if (k,l)!=(0,0) and (i+k)<ns and (j+l)<ms:
                            Threshold[i+k,j+l] = (Ta*(n-k)*(n-l) + Tc*k*(n-l) + Tb*l*(n-k) + Td*k*l) / n**2
    return( Threshold )


# =============================================================================
# Function that achieve Sauvola binarization
# =============================================================================

def Sauvola(image, n, r, R = 128, k = 0.5, interpolation = "SIMPLE"):
    """
    Algorithm specialized in text binarization. Achieve it with a modified
    Niblack's formula. n is a fast option is to compute first a threshold for 
    every nth pixel and then using interpolation for the rest of the pixels.

    Parameters
    ----------
    image : 2D array
        the image processed.
    n : INT
        option to compute a threshold every nth pixel.
    r : INT
        radius of the square considerated in the Niblack's method..
    R : INT, optional
        dynamic range of standard deviation. The default is 128.
    k : INT, optional
        parameter of the method. The default is 0.5.
    interpolation : STRING, optional
        the interpolation method chosen : "SIMPLE" or "BILINEAR"

    Returns
    -------
    Threshold : 2D array
        the matrix containing the threshold values
    result : 2D array
        the matrix representing the binarized image

    """
    
    ### Création d'une matrice 2D contenant les valeurs de Threshold et l'image finale
    ns, ms = np.shape(image)
    Threshold = np.zeros((ns, ms))
    result = np.zeros((ns, ms))
    
    ### Binarisation of the pixels on a n-step size
    for i in range(0, ns, n):
        for j in range(0, ms, n):
            Mean = Sauv_mean(image, i, j, r)
            Std = Sauv_standard_deviation(image, i, j, r, Mean)
            Th = Mean * (1 + k*(Std/R -1) )
            Threshold[i, j] = Th
            # print(i, j)
    
    ### Interpolation of the other pixels
    if n > 1:
        if interpolation == "SIMPLE":
            Threshold = interpol_simple(n, Threshold)
        else:
            Threshold = interpol_bili(n, image, Threshold)
    
    ### Calculation of the binarized imaged
    for i in range(ns):
        for j in range(ms):
            if image[i,j] >= Threshold[i,j]:
                result[i,j] = 1
    
    return(Threshold, result)

def Niblack(image, n, r):
    """ Méthode de Niblack pour binarisation"""
        ### Création d'une matrice 2D contenant les valeurs de Threshold et l'image finale
    ns, ms = np.shape(image)
    Threshold = np.zeros((ns, ms))
    result = np.zeros((ns, ms))
    
    ### Binarisation of the pixels on a n-step size
    for i in range(0, ns, n):
        for j in range(0, ms, n):
            Mean = Sauv_mean(image, i, j, r)
            Std = Sauv_standard_deviation(image, i, j, r, Mean)
            Th = Mean - 0.2*Std
            Threshold[i, j] = Th
            # print(i, j)
    
    ### Bilinear interpolation of the other pixels
    if n > 1:
        Threshold = interpol_bili(n, image, Threshold)
    
    ### Calculation of the binarized imaged
    for i in range(ns):
        for j in range(ms):
            if image[i,j] >= Threshold[i,j]:
                result[i,j] = 1
    
    return(Threshold, result)

# =============================================================================
# ### Use of the function ###
# =============================================================================

# Importation and normalization of the images
im1 = affiche.load_im("test_base2.png", grey = True)
im2 = affiche.load_im("test_lum.png", grey = True)
im3 = affiche.load_im("test_image.png", grey = True)

###############################################
image = im2
n = 3
r = 10
R = 128
k = 0.5

# t1 = time.perf_counter()
# # Sauvola binarization
# Threshold, result = Sauvola(image, n, r, R, k, interpolation = "BILINEAR")
# t = int(time.perf_counter() - t1)
# result *= 255

# # affiche.plot_curve(image, Threshold, 100)

# fig = affiche.plot_im([image, result], ["IMAGE", "BINARIZATION"], grey = True)
# fig.suptitle("Sauvola binarization : n = {}, r = {} / Temps d'exécution = {} sec".format(n, r, t), fontsize = 26, weight = 'bold')
# fig.savefig(os.path.join("results","result.jpg"))


###############################################
ima = [image]
legend =["Image"]
r = 7
n = 5
interp = ["SIMPLE", "BILINEAR"]

for inter in interp:
    
    t1 = time.perf_counter()
    # Sauvola binarization
    Threshold, result = Sauvola(image, n, r, R, k, interpolation = interp)
    t = int((time.perf_counter() - t1)*10)/10
    result *= 255
    
    ima.append(result)
    legend.append("Sauvola interp = {}\n T = {} sec".format(inter, t))

t1 = time.perf_counter()
# Niblacks binarization
Threshold, result = Niblack(image, n, r)
t = int((time.perf_counter() - t1)*10)/10
result *= 255

ima.append(result)
legend.append("Niblack interp = BILINEAR\n T = {} sec".format(t))    



fig = affiche.plot_im(ima, legend, grey = True)
fig.suptitle("binarizations", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("results","result.jpg"))
