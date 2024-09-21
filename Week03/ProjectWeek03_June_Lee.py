#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:20:43 2024

@author: junelee
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import time


"""
Problem 1
"""

# function to create an exponentially weighted covariance matrix
# input is a pandas M x N dataframe, where each of N columns is a data series 
# and each of M rows is time/entry,
# and exp_rate (lambda in the notes)
# output is the N x N exponentially weighted covariance matrix


def exp_weighted_cov_matrix(data, exp_rate):
    
    # make a copy of data
    df = data.copy()
    
    # create empty list to keep track of weights
    w = []
    m = len(df)
    tot_w = 0.0
    # for each row (timestep) starting at 1
    for i in range(1, m+1):
        
        # calculate w_(t-i)
        weight = (1-exp_rate)*(exp_rate**(i-1))
        
        # append to list
        # the list goes from i = 1 to i = m
        w.append(weight)
        
        # keep running sum of weights to normalize
        tot_w += weight
        
    # normalize the weights based on running sum
    for i in range(len(w)):
        w[i] = w[i]/tot_w
    
    # flip the list so that index of the list matches with the row index of 
    # initial data
    # k-th row entry data matches with k-th weight in list for final calculation
    w.reverse()
    
    # mean-center each column of returns data
    df = df.apply(lambda x: x-x.mean())
    
    # convert dataframe into numpy array for matrix multiplication (X^T)(X)
    # make a copy of data and transpose it
    # insert weights into X
    # matrix multiplication yields covariance matrix
    X_copy = df.copy().to_numpy()
    X = df.to_numpy()
    X_T = np.transpose(X_copy)
    
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j] = w[i]*X[i][j]
    
    
    cov = np.matmul(X_T,X)
    
    # return covariance matrix as a pandas DataFrame
    
    return(pd.DataFrame(cov))

"""
Problem 2
"""

# implementing chol_psd and and near_psd in python

def chol_psd(root, a):
    n = a.shape[0]
    # Initialize the root matrix with 0 values
    root.fill(0.0)

    # Loop over columns
    for j in range(n):
        
        s = 0.0
        # If we are not on the first column, calculate the dot product of the preceding row values.
        if j > 0:
            
            s = np.dot(root[j, :j], root[j, :j])


        # Diagonal Element
        temp = a[j, j] - s
        
        if 0 >= temp:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        # Check for the 0 eigenvalue. If the diagonal is zero, move to the next column
        if root[j, j] != 0.0:
            # Update off-diagonal rows of the column
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir
                
    return root


def near_psd(a, epsilon=0.0):
    n = a.shape[0]

    invSD = None
    out = np.copy(a)

    # Calculate the correlation matrix if we got a covariance
    if not np.isclose(np.diag(out), [1.0 for i in range(n)]).all:
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD

    # SVD, update the eigenvalues and scale
    vals, vecs = np.linalg.eigh(out)
    for i in range(len(vals)):
        vals[i] = np.maximum(vals[i], epsilon)
    
    # element-wise multiplication, transpose
    S = np.transpose(np.multiply(vecs, vecs))
    
    # take t_i values
    temp = vals @ S
    
    # initialize final list T with -1 so that any mistakes result in negative values
    T = [-1 for i in range(len(temp))]
    
    # update T values
    for i in range(len(T)):
        T[i] = 1/temp[i]
        
    # take diagonal matrix of the sqrt values
    T = np.diag(np.sqrt(T))
    B = T @ vecs @ np.diag(np.sqrt(vals))

    out = B @ B.T

    # Add back the variance
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD

    return out

def higham_near_psd(A, tol, maxits=500, w=None):
    
    if w == None:
        w = np.diag(np.ones(len(A)))

    # initialize variables
    # copy of input matrix A
    C = A.copy()
    Y = A.copy()
    # count number of iterations performed
    count = 0
    # difference to track
    diff = np.inf
    dS = np.zeros_like(A)
    new_norm = 0
    
    # while tolerance is not reached
    while diff > tol:
        # take previous timestep norm as prev_norm
        prev_norm = new_norm
        
        # algorithm using helper functions below
        R = Y - dS
        X = proj_spectral(R,w)
        dS = X - R
        Y = proj_u(X)
        
        
        # calculate new norm and diff, add to running counter
        new_norm = np.linalg.norm(Y - C, 'fro')
        diff = abs(new_norm-prev_norm)
        count += 1
        
        # if max iterations reached, break
        if count >= maxits:
            print("Reached Max Number of Iterations")
            return X, count
        
    return X, count

# helper functions
# second projection in notes
def proj_spectral(A,w):
    
    # weight matrix W^(1/2)
    W = np.matrix(sqrtm(w))
    
    # calculating new A to take (A)+
    A = W @ np.matrix(A) @ W
    vals, vecs = np.linalg.eigh(A)
    for i in range(len(vals)):
        vals[i] = np.maximum(vals[i],0.0)
    
    
    A = vecs @ np.diag(vals) @ vecs.T
    
    # take W^(-1/2) on both sides
    A = W.I @ A @ W.I
    return A

# first projection in notes
# assumes W is diagonal
def proj_u(A):
    n = len(A)
    for i in range(n):
        A[i,i] = 1
    return A


"""
Problem 3
"""

# function to simulate multivariate normal
# cov is covariance matrix to be used as input, uses numpy
# var_vec is the variance vector input, uses numpy
# pca=None by default, which is direct simulation
# pca = 1 uses PCA w 100%, pca = 0.75 uses 75%, etc
# n is integer number of draws to simulate

def multivariate_normal_sim(cov, n, mu, root, pca=None):
    
    # direct simulation
    if pca == None:
        
        # factor cov matrix into L
        L = chol_psd(root, cov)
        Z = np.random.standard_normal((len(L[0]),n))
        
        X = L @ Z
        for i in range(len(X)):
            for j in range(len(X[0])):
                X[i,j] += mu[i]
        
        X = X.transpose()
        return pd.DataFrame(X)
    
    elif pca != None and pca >= 0 and pca <= 1:
        # take eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eig(cov)
        
        # drop eigenvalues and corresponding eigenvectors smaller than 1/10^-8
        for i in range(len(eigvals)):
            if eigvals[i] < 1/(10**8):
                eigvals[i] = 0
                eigvecs[:,i] = 0
        
        
        # based on desired % explained, we want to drop additional eigenvalues
        # and corresponding vectors
        num_eigvals = 0
        tot = sum(eigvals)
        cum_explained = 0
        # iterate through eigvals from largest to greatest
        # stop once desired percentage is achieved
        for i in range(len(eigvals)):
            cum_explained += eigvals[i]/tot
            if cum_explained > pca:
                num_eigvals = i
                break
        
        # safety feature, if we do not break from loop, 
        # use all eigenvals/vectors
        if num_eigvals == 0:
            num_eigvals == len(eigvals) - 1
        
        new_eigvals = []
        new_eigvecs = np.zeros((len(eigvecs), num_eigvals + 1))
        # drop eigenvalues with index i+1 and greater
        for i in range(num_eigvals + 1):
            new_eigvals.append(eigvals[i])
            new_eigvecs[:,i] = eigvecs[:,i]
        
        
        for i in range(len(new_eigvals)):
            new_eigvals[i] = np.sqrt(new_eigvals[i])
        
        sq_eigvals = np.diag(new_eigvals)
        B = new_eigvecs @ sq_eigvals
        
        Z = np.random.standard_normal((len(B[0]),n))
        X = B @ Z
        
        for i in range(len(X)):
            for j in range(len(X[0])):
                X[i,j] += mu[i]
        
        X = X.transpose()
        return pd.DataFrame(X)
        
        
    else:
        raise ValueError("PCA parameter not in range")
    
    return

if __name__ == "__main__":
    
    
    """
    Problem 1
    """
    
    # get data
    data = pd.read_csv("DailyReturn.csv")
    
    
    # for a list of varying lambda values, calculate exp.-weighted cov matrix
    lambda_vals = [0.1 + 0.1*i for i in range(9)]
    
    # initialize empty list for plotting
    var_exp_general = []
    
    # for each lambda value,
    for exp_rate in lambda_vals:
        
        # calculate the exponentially weighted covariance matrix
        cov = exp_weighted_cov_matrix(data, exp_rate)
        
        # convert pd.DataFrame output to np.array
        cov1 = np.array(cov)
        
        # find eigenvalues and eigenvectors
        # automatically sorted from largest to smallest
        # eigenvalues are non-zero given PSD
        eig_vals, eig_vecs = np.linalg.eig(cov1)
        
        # take sum of all eigenvalues to find variance explained by each 
        # eigenvalue
        tot = sum(eig_vals)
        
        # initialize variables for plotting
        cum_var_exp = 0
        list_var_exp = []
        
        # create a list of cumulative variance explained by each eigenvalue
        for i in range(len(eig_vals)):
            cum_var_exp += eig_vals[i]/tot
            list_var_exp.append(cum_var_exp)
        
        # append to overall list for storage
        var_exp_general.append(list_var_exp)
        
    
    # plotting
    
    fig, ax = plt.subplots()

    for i in range(len(lambda_vals)):
        ax.plot([i for i in range(1,len(var_exp_general[0]) + 1)], var_exp_general[i],
                label=f"Lambda = {round(lambda_vals[i],1)}")

    plt.xlabel('Number of Eigenvalues')
    plt.ylabel('Cumulative Variance Explained')
    plt.legend()

    plt.show()
    
    
    
    """
    Problem 2
    """
    
    
    # Generate non-psd 500x500 matrix
    
    n = 500
    sigma = np.full((n, n), 0.9)
    for i in range(n):
        sigma[i,i] = 1.0

    sigma[0, 1] = 0.7357
    sigma[1, 0] = 0.7357
    
    # check non-PSD
    print(sigma)
    print(f"Min Eigenvalue: {min(np.linalg.eigvals(sigma))}")
    print("Original matrix is PSD: ")
    print(np.all(np.linalg.eigvals(sigma) >= 0))
    
    print("________________________________________________________")
    
    
    # near_psd()
    X = near_psd(sigma)

    print(X)
    print("near_psd output is PSD:")
    print(f"Min Eigenvalue: {min(np.linalg.eigvals(X))}")
    print(np.all(np.linalg.eigvals(X) >= 0))
    norm = np.linalg.norm(X - sigma, 'fro')
    print(f"norm = {norm}")
    
    print("________________________________________________________")
    
    # Higham's method
    tol = 1/10**16
    Y, z = higham_near_psd(sigma, tol)
    
    print(Y)
    print(f"Number of Iterations: {z}")
    print(f"Min Eigenvalue: {min(np.linalg.eigvals(Y))}")
    print("Higham's method output is PSD:")
    print(np.all(np.linalg.eigvals(Y) >= 0))
    norm1 = np.linalg.norm(Y - sigma, 'fro')
    print(f"norm = {norm1}")
    
    """
    Code below is used to generate plot for runtime
    
    
    list_n = [5,10,50,100,200,300,400,500]
    near_psd_rt = []
    higham = []
    
    for n in list_n:
        print(f"N = {n}")
        sigma = np.full((n, n), 0.9)
        for i in range(n):
            sigma[i,i] = 1.0

        sigma[0, 1] = 0.7357
        sigma[1, 0] = 0.7357
        
        print("Near_PSD")
        start_time = time.time()
        X = near_psd(sigma)
        near_psd_rt.append(time.time()-start_time)
        
        print("Higham's Method")
        start_time = time.time()
        Y = higham_near_psd(sigma, 1/10**16)
        higham.append(time.time()-start_time)
        
    
    near_psd1_rt = [ '%.2f' % elem for elem in near_psd_rt]
    higham1 = [ '%.2f' % elem for elem in higham ]
    print(near_psd1_rt)
    print(higham1)
    
    
    fig, ax = plt.subplots()

    ax.plot(list_n, near_psd_rt, label="near_psd() Run Time")
    ax.plot(list_n, higham, label="Higham's Method Run Time")

    plt.xlabel('N')
    plt.ylabel('Run Time (seconds)')
    plt.legend()

    plt.show()
    """
    
    
    """Problem 3"""
    
    # get data
    data = pd.read_csv("DailyReturn.csv")
    
    # number of series in data
    m = len(data.columns)
    
    # take sample mean
    mu = np.array(data.mean())
    
    ## Standard
    # pearson cov matrix
    cov = np.array(data.cov())
    # variance vector
    var_vec = np.array(data.var())
    
    ## Exp. Weighted
    # exp. weighted covariance matrix with lambda = 0.97
    exp_cov = np.array(exp_weighted_cov_matrix(data, 0.97))
    
    # take diagonals for variance vector
    exp_var_vec = np.diag(exp_cov)
    
    
    
    # initialize root for chol_psd()
    root = np.zeros((m,m))
    
    # number of draws
    n = 25000
    
    """
    cov1 = standard pearson covariance and standard variance vector
    cov2 = standard covariance and exp. weighted variance vector
    cov3 = exp. weighted covariance and standard var vector
    cov4 = exp. weighted covariance and exp. weighted var vector
    
    Note that standard covariance matrix already has standard variance 
    on the diagonal (case 1). Likewise for exp. weighted (case 4).
    """
    
    cov1 = cov.copy()
    cov2 = cov.copy()
    for i in range(len(cov2)):
        cov2[i,i] = exp_var_vec[i]
    cov3 = exp_cov.copy()
    for j in range(len(cov3)):
        cov3[i,i] = var_vec[i]
    cov4 = exp_cov.copy()
    
    # for final result presentation
    norm_direct = []
    norm_100 = []
    norm_75 = []
    norm_50 = []
    
    rt_direct = []
    rt_100 = []
    rt_75 = []
    rt_50 = []
    
    
    
    # cov1
    print("Input Covariance Matrix")
    print(pd.DataFrame(cov1))
    print("____________________________________________________")
    
    print("Direct Sim")
    start_time = time.time()
    A = multivariate_normal_sim(cov1, n, mu, root)
    rt_direct.append(time.time() - start_time)
    print(A.cov())
    norm_direct.append(np.linalg.norm(cov1 - A.cov(), 'fro'))
    print("____________________________________________________")
    
    print("PCA with 100%")
    start_time = time.time()
    A = multivariate_normal_sim(cov1, n, mu, root, pca = 1)
    rt_100.append(time.time() - start_time)
    print(A.cov())
    norm_100.append(np.linalg.norm(cov1 - A.cov(), 'fro'))
    print("____________________________________________________")
    
    print("PCA with 75%")
    start_time = time.time()
    A = multivariate_normal_sim(cov1, n, mu, root, pca = 0.75)
    rt_75.append(time.time() - start_time)
    print(A.cov())
    norm_75.append(np.linalg.norm(cov1 - A.cov(), 'fro'))
    print("____________________________________________________")
    
    print("PCA with 50%")
    start_time = time.time()
    A = multivariate_normal_sim(cov1, n, mu, root, pca = 0.5)
    rt_50.append(time.time() - start_time)
    print(A.cov())
    norm_50.append(np.linalg.norm(cov1 - A.cov(), 'fro'))
    print("____________________________________________________")
    
    
    # cov2
    print("Input Covariance Matrix")
    print(pd.DataFrame(cov2))
    print("____________________________________________________")
    
    print("Direct Sim")
    start_time = time.time()
    A = multivariate_normal_sim(cov2, n, mu, root)
    rt_direct.append(time.time() - start_time)
    print(A.cov())
    norm_direct.append(np.linalg.norm(cov2 - A.cov(), 'fro'))
    print("____________________________________________________")
    
    print("PCA with 100%")
    start_time = time.time()
    A = multivariate_normal_sim(cov2, n, mu, root, pca = 1)
    rt_100.append(time.time() - start_time)
    print(A.cov())
    norm_100.append(np.linalg.norm(cov2 - A.cov(), 'fro'))
    print("____________________________________________________")
    
    print("PCA with 75%")
    start_time = time.time()
    A = multivariate_normal_sim(cov2, n, mu, root, pca = 0.75)
    rt_75.append(time.time() - start_time)
    print(A.cov())
    norm_75.append(np.linalg.norm(cov2 - A.cov(), 'fro'))
    print("____________________________________________________")
    
    print("PCA with 50%")
    start_time = time.time()
    A = multivariate_normal_sim(cov2, n, mu, root, pca = 0.5)
    rt_50.append(time.time() - start_time)
    print(A.cov())
    norm_50.append(np.linalg.norm(cov2 - A.cov(), 'fro'))
    print("____________________________________________________")
    
    
    # cov3
    print("Input Covariance Matrix")
    print(pd.DataFrame(cov3))
    print("____________________________________________________")
    
    print("Direct Sim")
    start_time = time.time()
    A = multivariate_normal_sim(cov3, n, mu, root)
    rt_direct.append(time.time() - start_time)
    print(A.cov())
    norm_direct.append(np.linalg.norm(cov3 - A.cov(), 'fro'))
    print("____________________________________________________")
    
    print("PCA with 100%")
    start_time = time.time()
    A = multivariate_normal_sim(cov3, n, mu, root, pca = 1)
    rt_100.append(time.time() - start_time)
    print(A.cov())
    norm_100.append(np.linalg.norm(cov3 - A.cov(), 'fro'))
    print("____________________________________________________")
    
    print("PCA with 75%")
    start_time = time.time()
    A = multivariate_normal_sim(cov3, n, mu, root, pca = 0.75)
    rt_75.append(time.time() - start_time)
    print(A.cov())
    norm_75.append(np.linalg.norm(cov3 - A.cov(), 'fro'))
    print("____________________________________________________")
    
    print("PCA with 50%")
    start_time = time.time()
    A = multivariate_normal_sim(cov3, n, mu, root, pca = 0.5)
    rt_50.append(time.time() - start_time)
    print(A.cov())
    norm_50.append(np.linalg.norm(cov3 - A.cov(), 'fro'))
    print("____________________________________________________")
    
    
    # cov4
    print("Input Covariance Matrix")
    print(pd.DataFrame(cov4))
    print("____________________________________________________")
    
    print("Direct Sim")
    start_time = time.time()
    A = multivariate_normal_sim(cov4, n, mu, root)
    rt_direct.append(time.time() - start_time)
    print(A.cov())
    norm_direct.append(np.linalg.norm(cov4 - A.cov(), 'fro'))
    print("____________________________________________________")
    
    print("PCA with 100%")
    start_time = time.time()
    A = multivariate_normal_sim(cov4, n, mu, root, pca = 1)
    rt_100.append(time.time() - start_time)
    print(A.cov())
    norm_100.append(np.linalg.norm(cov4 - A.cov(), 'fro'))
    print("____________________________________________________")
    
    print("PCA with 75%")
    start_time = time.time()
    A = multivariate_normal_sim(cov4, n, mu, root, pca = 0.75)
    rt_75.append(time.time() - start_time)
    print(A.cov())
    norm_75.append(np.linalg.norm(cov4 - A.cov(), 'fro'))
    print("____________________________________________________")
    
    print("PCA with 50%")
    start_time = time.time()
    A = multivariate_normal_sim(cov4, n, mu, root, pca = 0.5)
    rt_50.append(time.time() - start_time)
    print(A.cov())
    norm_50.append(np.linalg.norm(cov4 - A.cov(), 'fro'))
    print("____________________________________________________")

    
    norm = pd.DataFrame([norm_direct, norm_100, norm_75, norm_50],
                        index=["Direct", "100%", "75%", "50%"],
                        columns=["A","B","C","D"])
    
    rt = pd.DataFrame([rt_direct, rt_100, rt_75, rt_50],
                        index=["Direct", "100%", "75%", "50%"],
                        columns=["A","B","C","D"])
    
    
    print("Frobenius Norm")
    print(norm)
    print("Runtime")
    print(rt)
    
    