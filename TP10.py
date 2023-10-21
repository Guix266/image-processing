# -*- coding: utf-8 -*-
"""
Created on Tue May 19 08:44:47 2020

@author: guix
TP 10 : multiscale analysis (12)
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import scipy.ndimage
import scipy
from PIL import Image
from skimage import morphology

import img_funct

im1 = img_funct.load_im("cerveau.jpg", grey = True)

# =============================================================================
# 12.1 Pyramidal decomposition and reconstruction
# =============================================================================
#%%

def Pyramid(image):
    """ Construct the pyramid of approximation 4 and
    The the pyramid of detail associated """
    
    pyr_approx = []
    pyr_detail = []
    
    sigma = 3
    for i in range(4):
        prevImage = image.copy()
        
        # Gaussian filtering
        g = scipy.ndimage.gaussian_filter(image, sigma)
        
        # Downsampling
        width, height = np.shape(image)
        image = np.array(Image.fromarray(g).resize((int(width/2), int(height/2)), resample = Image.BILINEAR))
        primeImage = np.array(Image.fromarray(image).resize((width, height), resample = Image.BILINEAR))
                            
        # Calculation of the details 
        pyr_approx.append(prevImage)
        pyr_detail.append(prevImage - primeImage)
    
    pyr_approx.append(image)
    pyr_detail.append(image)
    return(pyr_approx, pyr_detail)

# pyr_approx, pyr_detail = Pyramid(im1)
# fig = img_funct.plot_im(pyr_approx, binary = True)
# fig = img_funct.plot_im(pyr_detail, binary = True)

def reconstruct(pyr_approx, pyr_detail):
    """ Reconstruct the image"""
    
    image = pyr_approx[- 1]
    width, height = np.shape(image)    
    for i in range(len(pyr_approx)-2, -1, -1):
        width, height = np.shape(pyr_approx[i])   
        # image = pyr_approx[i] + np.array(Image.fromarray(image).resize((width, height)))
        image = pyr_approx[i] + pyr_detail[i]
    return(image)

# im = reconstruct(pyr_approx, pyr_detail)   
# img_funct.plot_im([im1, im], binary = True)


# =============================================================================
# 12.2 morphoMultiscale
# =============================================================================


def morphoMultiscale(I , levels) :
    """
    Morphological multiscale decomposition
    I : original image, 
    oat32
    levels : number of levels , int
    returns : pyrD, pyrE: pyramid of Dilations /Erosions, respectively
    """
    pyrD=[]
    pyrE=[]
    for r in np.arange(1, levels):
        se = morphology.disk(r)
        pyrD.append( morphology.dilation ( I , selem=se))
        pyrE.append( morphology.erosion(I , selem=se))
    return(pyrD, pyrE)

# pyr_approx, pyr_detail = morphoMultiscale(im1, 5)
# fig = img_funct.plot_im(pyr_approx, binary = True)
# fig = img_funct.plot_im(pyr_detail, binary = True)


# =============================================================================
# 12.3 Kramer and Bruckner multiscale decomposition
# =============================================================================

def kb(I , r) :
    """
    Elementary Kramer/Bruckner lter . Also called toggle lter .
    I : image
    r: radius of structuring element ( disk ) , for max/min evaluation
    """
    se = morphology.disk(r)
    width, height = np.shape(I)
    D = morphology.dilation(I, selem=se)
    E = morphology.erosion(I, selem=se)
    new = I.copy()
    
    for i in range(width):
        for j in range(height):
            if (D[i,j]-I[i,j] < I[i,j]-E[i,j]):
                new[i,j] = D[i,j]
            else:
                new[i,j] = E[i,j]
    return(I)

def KBmultiscale( I , levels , r=1) :
    """
    Kramer and Bruckner multiscale decomposition
    I : original image, oat32
    pyrD: pyramid of Dilations
    pyrE: pyramid of Erosions
    returns : MKB: Kramer/Bruckner filters
    """
    MKB = []
    MKB.append(I)
    for i in range(levels):
        MKB.append(kb(MKB[i], r))
    return MKB

im = KBmultiscale(im1, 5, r=5)
fig = img_funct.plot_im(im, binary = True)

