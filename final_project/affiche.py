# -*- coding: utf-8 -*-
"""
Created on Sun May 10 18:40:04 2020

@author: Guillaume DESERMEAUX
Mini-projet : Sauvola Binarization
"""
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os

# =============================================================================
# Import an image
# =============================================================================

def load_im(image, folder = "", grey = False):
    """Load an image and stock it (add folder inside of the 'images' folder)"""
    folder = os.path.join("images",folder)
    image = img.imread(os.path.join(folder,image))
    # print(np.shape(image))
    if grey:
        print("[INFO] : Grey image imported")
        if (len(np.shape(image)) >= 3):
            print("[INFO] : Image converted to grey scale")
            image = np.dot(image[...,:3], [0.299, 0.587, 0.144])
            
            # Normalisation of the image
            if np.max(image) <=10:
                M = np.max(image)
                m = np.min(image)
                image = 255 *(image-m)/(M-m)
            image = image.astype(int)
    return(image)


# print(load_im("bird-1.bmp","images_Kimia216"))
# a = load_im("retine.png", grey=True)
# print(np.shape(a))
# load_im("cornee.png", grey=True)

# =============================================================================
# Print images
# =============================================================================

def plot_im(images, legend = 0, grey = False):
    """plot the images in a window.
    The images are in a list"""
    i = len(images)
    fig, axs = plt.subplots(nrows = 2, ncols = i//2, figsize = (30,30))
    for j in range(i):
        ax = plt.subplot(2, i//2 , j+1)
        ax.set_xticks([])
        ax.set_yticks([])
        if type(legend) == list:
            if len(legend) == i:
                plt.title(legend[j], fontsize=30)
            else:
                return("Error: Wrong number of elements in the legend")
        if grey:
            plt.imshow(images[j],  cmap = 'gray')
        else:
            plt.imshow(images[j])
    plt.show()
    return(fig)
    

# im1 = load_im("retine.png")
# im2 = load_im("muscle.jpg")
# im3 = load_im("cellules_cornee.jpg")
# plot_im([im1, im2, im3, im1],["legend", "legend", "legend", "legend"])

# =============================================================================
# Plot Threshold value and grey level value curve
# =============================================================================

def plot_curve(image, Threshold, l):
    """ plot the curve if the image has a wide>l"""
    n, m = np.shape(image)
    if m<=l:
        raise ValueError('image uncompatible. Must have bigger wide !')
    
    i = int(n/2)
    j=0
    x = []
    th = []
    im = []
    while (len(th) <= l) :
        th.append(Threshold[i,j])
        im.append(image[i,j])
        x.append(j)
        j += 1
    print(x)
    print(th)
    plt.plot(x,th, label = 'Threshold line')
    plt.plot(x,im, label = 'image line')
    plt.legend()
    plt.show()


# =============================================================================
# Make images of 256 octets
# =============================================================================

def Normalize(image):
    M = np.max(image)
    m = np.min(image)
    image = 255 *(image-m)/(M-m)
    image = image.astype(int)
    return(image)


# =============================================================================
# Redimension des images pour avoir une plus petite taille 
# =============================================================================

# im = Image.open(os.path.join("images","test_image.jpg"))
# n, m = im.size
# n = int(n/6)
# m = int(m/6)
# im = im.resize((n, m), Image.ANTIALIAS)
# im.save(os.path.join("images","test_image.png"), "PNG")

