# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:37:05 2021

@author: htchen
"""
# If this script is not run under spyder IDE, comment the following two lines.
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import cv2

# calculate the eigenvalues and eigenvectors of a squared matrix
# the eigenvalues are decreasing ordered
def myeig(A):
    lambdas, V = np.linalg.eig(A)
    # lambdas, V may contain complex value
    lambdas, V = np.real(lambdas), np.real(V)
    lambda_abs = np.abs(lambdas)
    idx = lambda_abs.argsort()[::-1]   
    return lambdas[idx], V[:,idx]

# SVD: A = U * Sigma * V^T
# V: eigenvector matrix of A^T * A; U: eigenvector matrix of A * A^T 
def mysvd(A):
    lambdas, V = myeig(A.T @ A)
    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))
    return U, Sigma, V
    
# 讀取影像檔, 並保留亮度成分
img = cv2.imread('data/svd_demo2.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

# convert img to float data type
A = img.astype(dtype=np.float64)

# SVD of A
U, Sigma, V = mysvd(A)
VT = V.T
# get sum of the first r comonents of SVD
r = 40
A_bar = U[:, 0:r] @ Sigma[0:r, 0:r] @ VT[0:r, :]

plt.imshow(A_bar, cmap='gray', vmin = 0, vmax = 255)
plt.axis('off')
plt.show()

img_h, img_w = A.shape
# Compute SNR
keep_r = 200
snrs = np.zeros(keep_r)
ratios = np.zeros(keep_r)
Energy_A = np.sum(A * A)
for r in range(1, keep_r + 1):
    A_bar = U[:, 0:r] @ Sigma[0:r, 0:r] @ VT[0:r, :]
    Noise = A - A_bar
    Energy_N = np.sum(Noise * Noise)
    snrs[r - 1] = 10.0 * math.log10(Energy_A / Energy_N)
    ratios[r - 1] = (img_h * r + r + r * img_w) / (img_h * img_w)
    
plt.plot(np.arange(1, keep_r + 1), snrs, c='r')
plt.show()

plt.plot(np.arange(1, keep_r + 1), ratios, c='r')
plt.show()