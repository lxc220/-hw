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

plt.rcParams['figure.dpi'] = 144 

# calculate the eigenvalues and eigenvectors of a squared matrix
# the eigenvalues are decreasing ordered
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    # lambdas, V may contain complex value
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1] 
    return lambdas[sorted_idx], V[:, sorted_idx]

# SVD: A = U * Sigma * V^T
# V: eigenvector matrix of A^T * A; U: eigenvector matrix of A * A^T 
def mysvd(A):
    lambdas, V = myeig(A.T @ A, symmetric=True)
    lambdas, V = np.real(lambdas), np.real(V)
    # if A is full rank, no lambda value is less than 1e-6 
    # append a small value to stop rank check
    lambdas = np.append(lambdas, 1e-12)
    rank = np.argwhere(lambdas < 1e-6).min()
    lambdas, V = lambdas[0:rank], V[:, 0:rank]
    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))
    return U, Sigma, V

# 讀取影像檔, 並保留亮度成分
img = cv2.imread('data/svd_demo1.jpg', cv2.IMREAD_GRAYSCALE)

# convert img to float data type
A = img.astype(dtype=np.float64)

# SVD of A
U, Sigma, V = mysvd(A)
VT = V.T


def compute_energy(X: np.ndarray):
    return np.sum(X**2)
    
    
# img_h and img_w are image's height and width, respectively
img_h, img_w = A.shape
# Compute SNR
keep_r = 201
rs = np.arange(1, keep_r)


# compute energy of A, and save it to variable Energy_A
energy_A = compute_energy(A)

# Decalre an array to save the energy of noise vs r.
# energy_N[r] is the energy of A - A_bar(sum of the first r components)
energy_N = np.zeros(keep_r) # energy_N[0]棄置不用

for r in rs:
    # A_bar is the sum of the first r comonents of SVD
    # A_bar is an approximation of A
    A_bar = U[:, 0:r] @ Sigma[0:r, 0:r] @ VT[0:r, :] 
    Noise = A - A_bar 
    energy_N[r] = compute_energy(Noise) 

# 計算snr和作圖
# write your code here

snr = 10 * np.log10(energy_A / energy_N[1:])

plt.plot(rs, snr, label='SNR vs r')
plt.xlabel('r')
plt.ylabel('SNR (dB)')
plt.legend()
plt.show()

# 驗證能量
lambda_values = np.diag(Sigma)**2
for r in rs:
    energy_diff = np.sum(lambda_values[r:])
    assert np.isclose(energy_diff, energy_N[r]), f"Mismatch at r={r}"
    

# --------------------------
# verify that energy_N[r] equals the sum of lambda_i, i from r+1 to i=n,
# lambda_i is the eigenvalue of A^T @ A
# write your code here
