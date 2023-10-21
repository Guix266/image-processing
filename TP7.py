# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:16:24 2020

@author: guix
TP7 : Segmentation by region growing (20)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import queue

import img_funct

im1 = img_funct.load_im("cells.bmp")

# =============================================================================
# 19.1 Transformation estimation
# =============================================================================
#%%

# start by displaying a figure ,
# ask for mouse input ( click )
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111)
ax.set_title("Click on a point")
# load image
img = misc.ascent()
ax.imshow(img, picker = True, cmap = plt.gray () )


def predicate(image, i, j, seed):
    return( abs(image[i,j] - image[seed[0], seed[1]])<= 20)


def onpick(event) :
    """ Function that use the growing algorithm """
    print("click: xdata=%f, ydata=%f" % (event.xdata, event.ydata))
    
    # original pixel
    seed = np.array([int(event.ydata), int(event.xdata)])
    q=queue.Queue()
    
    # initializes the queue
    q.put(seed)
    
    
    # this matrix will contain 1 if in the region , −1 if visited but not in the
    #  region , 0 if not visited
    visited = np.ones(img.shape)
    #−−−−−−−−−−−−−−−
    # Start of algorithm
    visited [seed [0], seed [1]] = 1;
    
    while not q.empty():
        p = q.get()
        for i in range(max(0,p[0]-1), min(img.shape[0], p[0] + 2)):
            for j in range(max(0,p[1]-1), min(img.shape[1], p[1] + 2)):
                if visited[ i , j ] == 1:
                    if predicate(img, i , j , seed) :
                        visited[i,j] = 0
                        q.put (np.array ([ i , j ]) )
                    else :
                        visited[i, j] = -1
    # visited matrix contains the segmentation result
    # display results
    # ax = fig.add_subplot(122)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if visited[i, j] == -1:
                visited[i, j] = 1
    
    zone = visited*img
    print(zone)
    ax.imshow(zone)
    
    fig.canvas.draw ()
    plt.show()
    
    
# connect click on image to onpick function
cid = fig.canvas.mpl_connect("button_press_event" , onpick)
plt.show()




