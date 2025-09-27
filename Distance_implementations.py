# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 09:34:03 2025

@author: parse pardaz
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import math



def gaussian_kernel(x, y, sigma=1.0):
    x = x[:, np.newaxis] if x.ndim == 1 else x
    y = y[:, np.newaxis] if y.ndim == 1 else y
    dist = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
    return np.exp(-dist / (2 * sigma ** 2))

def polynomial_kernel(x, y, degree=3, coef0=1):
    x=x.reshape(len(x),1)
    y=y.reshape(1,len(y))
    return (x @ y + coef0) ** degree

def compute_mmd(X, Y, kernel='rbf', **kwargs):
    if kernel == 'rbf':
        K_xx = gaussian_kernel(X, X, **kwargs)
        K_yy = gaussian_kernel(Y, Y, **kwargs)
        K_xy = gaussian_kernel(X, Y, **kwargs)
    elif kernel == 'poly':
        K_xx = polynomial_kernel(X, X, **kwargs)
        K_yy = polynomial_kernel(Y, Y, **kwargs)
        K_xy = polynomial_kernel(X, Y, **kwargs)
    else:
        raise ValueError("Unknown kernel type. Use 'rbf' or 'poly'.")

    m = len(X)
    n = len(Y)
    
    # mmd_sq = (np.sum(K_xx) - np.trace(K_xx)) / (m * (m - 1)) \
    #        + (np.sum(K_yy) - np.trace(K_yy)) / (n * (n - 1)) \
    #        - 2 * np.sum(K_xy) / (m * n)
    mmd_sq = (np.sum(K_xx)) / (m * (m )) \
           + (np.sum(K_yy)) / (n * (n )) \
           - 2 * np.sum(K_xy) / (m * n)
    
    return  mmd_sq#np.sqrt(mmd_sq)
 


def compute_mmd_gpu(X, Y, n, b,kernel='rbf', **kwargs):
    s=0
    for i in range(b):
        s=s+compute_mmd(X[i,:], Y[i,:], kernel, **kwargs)
    return s
    
'''usage 

np.random.seed(0)
X = np.random.normal(0, 1, (100, 5))     # Sample from N(0, I)
Y = np.random.normal(0.5, 1, (100, 5))   # Sample from N(0.5, I)

# Gaussian kernel
mmd_rbf = compute_mmd(X, Y, kernel='rbf', sigma=1.0)

# Polynomial kernel
mmd_poly = compute_mmd(X, Y, kernel='poly', degree=3, coef0=1)

print(f"MMD (Gaussian): {mmd_rbf:.4f}")
print(f"MMD (Polynomial): {mmd_poly:.4f}")



X = np.random.normal(0, 1, (121, 1))     # Sample from N(0, I)
Y = np.random.normal(0.5, 1, (121, 1))   # Sample from N(0.5, I)

# Gaussian kernel
mmd_rbf = compute_mmd(X, Y, kernel='rbf', sigma=1.0)
mmd_poly = compute_mmd(X, Y, kernel='poly', degree=3, coef0=1)
print(mmd_poly )
'''








# def frechet_distance(X, Y, eps=1e-6):
#     """
#     Compute the Fréchet Distance between two datasets X and Y.
#     Each of shape (N, n), where N is number of samples and n is feature dimension.
#     """
#     X = np.asarray(X)
#     Y = np.asarray(Y)
    
#     mu_X = np.mean(X, axis=0)
#     mu_Y = np.mean(Y, axis=0)
    
#     sigma_X = np.cov(X, rowvar=False)
#     sigma_Y = np.cov(Y, rowvar=False)

#     # Product might not be symmetric due to numerical error, force symmetry
#     covmean = sqrtm(sigma_X @ sigma_Y)
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real

#     # Numerical error can make matrix nearly singular
#     if not np.all(np.isfinite(covmean)):
#         print("Adding epsilon to diagonals for numerical stability.")
#         sigma_X += np.eye(sigma_X.shape[0]) * eps
#         sigma_Y += np.eye(sigma_Y.shape[0]) * eps
#         covmean = sqrtm(sigma_X @ sigma_Y)
#         if np.iscomplexobj(covmean):
#             covmean = covmean.real

#     diff = mu_X - mu_Y
#     fid = diff @ diff + np.trace(sigma_X + sigma_Y - 2 * covmean)
#     return fid

'''usage

# Suppose each dataset has 100 samples, each of dimension 5
X = np.array([[i, i+1, i+2, i+3, i+4] for i in range(100)])  # shape (100, 5)
Y = np.array([[i+1, i+2, i+3, i+4, i+5] for i in range(100)])  # shifted version

fid_value = frechet_distance(X, Y)
print(f"Fréchet Distance: {fid_value:.4f}")


'''




# def matrix_sqrt(A, eps=1e-10):
#     eigvals, eigvecs = np.linalg.eigh(A)
#     eigvals_clipped = np.clip(eigvals, a_min=eps, a_max=None)
#     sqrt_eigvals = np.sqrt(eigvals_clipped)
#     return eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T

# def frechet_distance_numpy(X, Y, eps=1e-20):
#     X = np.asarray(X)
#     Y = np.asarray(Y)

#     mu_X = np.mean(X, axis=0)
#     mu_Y = np.mean(Y, axis=0)

#     cov_X = np.cov(X, rowvar=False)
#     cov_Y = np.cov(Y, rowvar=False)

#     cov_prod_sqrt = matrix_sqrt(cov_X @ cov_Y, eps=eps)

#     diff = mu_X - mu_Y
#     fid = diff @ diff + np.trace(cov_X + cov_Y - 2 * cov_prod_sqrt)
#     return fid



def matrix_sqrt(A, eps=1e-10):
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

def frechet_distance_numpy(X, Y, eps=1e-10):
    X = np.asarray(X)
    Y = np.asarray(Y)

    mu_X = np.mean(X, axis=0)
    mu_Y = np.mean(Y, axis=0)
    

    cov_X = np.cov(X, rowvar=False)
    cov_Y = np.cov(Y, rowvar=False)

    sqrt_cov_X = matrix_sqrt(cov_X, eps)
    mid_matrix = sqrt_cov_X @ cov_Y @ sqrt_cov_X
    cov_prod_sqrt = matrix_sqrt(mid_matrix, eps)

    diff = mu_X - mu_Y
    fid = diff @ diff + np.trace(cov_X + cov_Y - 2 * cov_prod_sqrt)
    return fid



###############################################################################
def calculate_hamming_distance(matrix1, matrix2):
    """
    Calculate Hamming distance between corresponding rows of two binary matrices.
    
    Args:
        matrix1: First binary matrix of shape (2048, 121)
        matrix2: Second binary matrix of shape (2048, 121)
        
    Returns:
        A numpy array of shape (2048, 1) containing Hamming distances
    """
    # Ensure inputs are numpy arrays
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    matrix1_int = matrix1.astype(int)
    matrix2_int = matrix2.astype(int)
    matrix_row,matrix_column=matrix1.shape
    # print(matrix_1_int.dtype)
    
    # Verify matrix shapes
    # if matrix1.shape != (2048, 121) or matrix2.shape != (2048, 121):
    #     raise ValueError("Both input matrices must be of shape (2048, 121)")
    
    # Calculate XOR (1 where elements differ, 0 where same)
    xor_result = np.bitwise_xor(matrix1_int, matrix2_int)
    
    
    # Sum along rows to get Hamming distance for each row
    hamming_distances = np.sum(xor_result, axis=1, keepdims=True)
    
    return hamming_distances 

    


####Polar Decodwer 
def bpsk_modulate(bits):
    return 1 - 2 * bits  # 0 → +1, 1 → -1

def awgn_channel(x, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    sigma = np.sqrt(1 / (2 * snr_linear))
    noise = sigma * np.random.randn(*x.shape)
    return x + noise

def compute_llr(received, snr_db):
    snr_linear = 10 ** (snr_db / 10)
    sigma2 = 1 / (2 * snr_linear)
    return 2 * received / sigma2

def belief_propagation(H, llr_input, max_iter=50):
    """
    Belief Propagation decoder for binary linear code.
    Inputs:
        H: parity-check matrix of shape (m, n)
        llr_input: initial LLRs from channel of shape (n,)
    Returns:
        decoded bits of shape (n,)
    """
    m, n = H.shape
    # Variable node to check node messages
    v2c = np.tile(llr_input, (m, 1))  # shape (m, n)
    v2c *= H

    for it in range(max_iter):
        # Check node to variable node messages
        c2v = np.zeros((m, n))

        for i in range(m):
            indices = np.where(H[i])[0]
            for j in indices:
                others = np.delete(indices, np.where(indices == j))
                if len(others) == 0:
                    c2v[i, j] = 0
                else:
                    tanh_prod = np.prod(np.tanh(0.5 * v2c[i, others]))
                    c2v[i, j] = 2 * np.arctanh(tanh_prod)

        # Update v2c messages
        for j in range(n):
            indices = np.where(H[:, j])[0]
            for i in indices:
                others = np.delete(indices, np.where(indices == i))
                v2c[i, j] = llr_input[j] + np.sum(c2v[others, j])

        # Decision
        total_llr = llr_input + np.sum(c2v, axis=0)
        x_hat = (total_llr < 0).astype(int)

        # Check syndrome
        if np.all(np.mod(H @ x_hat, 2) == 0):
            break

    return x_hat


# Define Polar Code Parameters
# n = 64  # blocklength
# k = 32  # message length

# EbNo_range_test = range(4, 7)
# std_test = [2*(k/n)*ii for ii in EbNo_range_test]
# snr_db_list=[10*math.log10(ii) for ii in std_test ]

# Generate polar code generator and parity check matrix (or load from file)
# Here we assume you have G and H already 
# G: k x n, H: (n-k) x n

# For testing: simple polar code

# G1 = torch.load('G_POLAR64,32.pt').numpy().astype(int)  # Shape (64,32)
# H1 = torch.load('H_POLAR64,32.pt').numpy().astype(int)  # Shape (32,64)
# G=np.transpose(G1)
# H=np.transpose(H1)

# Usage
# for ii in range(3):
    
#     u = np.random.randint(0, 2, size=(k,))
#     x = np.mod(u @ G, 2)

# # Transmit over AWGN
#     x_mod = bpsk_modulate(x)
#     y = awgn_channel(x_mod, snr_db=2)
#     llr = compute_llr(y, snr_db=2.0)

# # Decode
#     decoded = belief_propagation(H, llr, max_iter=100)


#     num_bit_errors = np.sum(x != decoded)

# # Print
# print("Original:", x)
# print("Decoded :", decoded)
# print(num_bit_errors)





