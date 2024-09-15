#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:46:44 2024

@author: junelee
"""

import numpy as np
import pandas as pd
from scipy import stats
import scipy
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

"""
# Problem 1
"""

# manually calculating first four moments using normalized formulas
# input: pandas dataframe
# output: mean, unbiased variance, skew, kurt
def calc_moments_manual(data):
    
    #calculate number of observations
    n = len(data)
    
    # find sample mean by taking sum of data and dividing by n
    sample_mean = data.sum()/n

    # initialize variables for var, skew, kurt
    sample_var = 0
    sample_skew = 0
    sample_kurt = 0
    
    # iterate through data to calculate summation
    for i in range(n):
        sample_var += (data.iloc[i]-sample_mean)**2
        sample_skew += (data.iloc[i]-sample_mean)**3
        sample_kurt += (data.iloc[i]-sample_mean)**4
        
    # given summation from above, adjust for final value
    sample_var = sample_var/(n-1)
    sample_skew = n*sample_skew/((n-1)*(n-2)*(sample_var**(3/2)))
    
    # set up calculation for unbiased kurt
    a = (n+1)*n/((n-1)*(n-2)*(n-3))
    
    # calculate final value for kurt
    sample_kurt = a*sample_kurt/(sample_var**2) - 3*((n-1)**2)/((n-2)*(n-3))
    
    return(float(sample_mean), float(sample_var), float(sample_skew), float(sample_kurt))
    
    
"""    
# Problem 2
"""

# simple linear regression using OLS
# input: pandas dataframe, independent variable as x and dependent as y
# output: OLS output

def simpleOLS(data):
    data = pd.read_csv("problem2.csv")
    x = data["x"]
    y = data["y"]
    return stats.linregress(x,y)

# log-likelihood function for multiple linear regression
# input params = [list of betas, last entry is sigma]
# X is np.array of observed data
# y dependent variable
# take the negative of the log likelihood and minimize it
def norm_log_likelihood(params, X, y):
    # take parameters, first n-1 parameters as beta, last one as sigma
    beta = params[:-1]
    sigma = params[-1]
    
    # predicted values using matrix multiplication
    y_pred = X @ beta
    # calculate error terms based on matrix multiplication
    err = y - y_pred
    # normal log-likelihood, return negative and minimize
    n = len(y)
    log_likelihood = -n/2 * np.log(2 * np.pi * sigma**2) - np.sum(err**2) / (2 * sigma**2)
    return -log_likelihood

def t_log_likelihood(params, X, y):
    # take parameters, first n-2 as beta, n-1th as nu, last one as sigma
    beta = params[:-2]
    nu = params[-2]  
    sigma = params[-1]
    
    # predicted values using matrix multiplication
    y_pred = X @ beta
    # calculate error terms based on matrix multiplication
    err = y - y_pred
    
    # 
    # t-distribution log-likelihood, return negative and minimize
    log_likelihood = np.sum(stats.t.logpdf(err / sigma, df=nu)) - len(y) * np.log(sigma)
    return -log_likelihood



if __name__ == "__main__":
    
    """
    Problem 1
    """
    
    
    # import data for problem 1 as pandas dataframe
    data = pd.read_csv("problem1.csv")
    mean, var, skew, kurt = calc_moments_manual(data)
    
    print("Part 1")
    print(f"Mean = {mean}")
    print(f"Variance = {var}")
    print(f"Skew = {skew}")
    print(f"Kurtosis = {kurt}")
    print("------------------------------------------")
    
    print("Part 2")
    mean1 = float(data.mean())
    var1 = float(data.var())
    skew1 = float(data.skew())
    kurt1 = float(data.kurt())
    print(f"Mean = {mean1}")
    print(f"Variance = {var1}")
    print(f"Skew = {skew1}")
    print(f"Kurtosis = {kurt1}")
    print("------------------------------------------")
    
    print("Part 3")
    test = pd.DataFrame([0,0,2,6])
    mean2 = float(test.sum()/len(test))
    var2 = float(test.var())
    skew2 = float(test.skew())
    kurt2 = float(test.kurt())
    print(f"Mean = {mean2}")
    print(f"Variance = {var2}")
    print(f"Skew = {skew2}")
    print(f"Kurtosis = {kurt2}")
    
    
    
    
    """
    Problem 2
    """
    
    
    data1 = pd.read_csv("problem2.csv")
    x = data1["x"]
    y = data1["y"]
    
    # OLS
    results = stats.linregress(x,y)
    print("OLS Results")
    print(results)
    print("------------------------------------------")
    # calculating the std err of error terms
    errors = []
    n = len(data1)
    # for each entry, take actual y - predicted y
    for i in range(n):
        epsilon = y.iloc[i]-(results.intercept + results.slope*x.iloc[i])
        errors.append(epsilon)
    
    # enter error terms into a pandas dataframe
    errors = pd.DataFrame(errors)
    
    # take the variance of the errors, 
    # multiplying by n-1/n to get biased estimate
    
    print(f"Biased Std. Err. from OLS = {float((errors.var()*((n-1)/n))**(1/2))}")
    print("------------------------------------------")


    # MLE estimation, normal distribution
    # data as np.array for matrix multiplication
    X = np.array(x).reshape((len(x),1))
    y = np.array(y)
    
    # add a column of ones to X for intercept
    const = np.ones((X.shape[0], 1))
    X_const = np.hstack((const,X))
    
    
    # initialize beta and sigma
    # beta_0 through beta_n, where n = number of columns in new X_const array
    # append sigma (initial guess of 1)
    initial = np.append(np.zeros(X_const.shape[1]), 1.0)

    # minimize the negative log-likelihood function
    result = scipy.optimize.minimize(norm_log_likelihood, initial, 
                                     args=(X_const, y), 
                                     bounds=[(None, None)] * (X_const.shape[1]) + [(0.001, None)])

    # print the estimated parameters
    estimated_params = result.x
    beta_est = estimated_params[:-1]
    sigma_est = estimated_params[-1]

    print(f"Estimated Beta coefficients MLE: {beta_est}")
    print(f"Estimated std err MLE: {sigma_est}")
    
    # MLE estimation, T dist
    # initialize beta, nu, and sigma
    initial_t = np.append(np.zeros(X_const.shape[1]), [5.0, 1.0])

    # Minimize the negative log-likelihood function
    result = scipy.optimize.minimize(t_log_likelihood, initial_t, args=(X_const, y),
                      bounds=[(None, None)] * (X_const.shape[1]) + [(2.0, None), (0.001, None)])

    # print the estimated parameters
    estimated_params = result.x
    beta_est = estimated_params[:-2]
    nu_est = estimated_params[-2]
    sigma_est = estimated_params[-1]
    print("------------------------------------------")
    print(f"Estimated coefficients: {beta_est}")
    print(f"Estimated degrees of freedom (nu): {nu_est}")
    print(f"Estimated scale parameter (sigma): {sigma_est}")
    print("------------------------------------------")
    
    
    # fitting a multivariate distribution
    data2 = pd.read_csv("problem2_x.csv")
    
    #print mean and covariance matrix
    print(data2.mean())
    print(data2.cov())
    
    # set mean and covariance matrix to variable
    mean = data2.mean()
    cov = data2.cov()
    
    # create empty list for iteration
    expected_x2 = []
    upper = []
    lower = []
    # calculate var_x2 based on sample
    var_x2 = cov.iloc[1,1] - cov.iloc[0,1]**2/cov.iloc[0,0]
    # iterate through the data, find predicted x_2 value for each observation
    # create list of upper and lower bound of the 95% CI for plotting
    for i in range(len(data2)):
        
        pred_x2 = mean.iloc[1] - (cov.iloc[0,1])*(data2.iloc[i,0]-mean.iloc[0])/cov.iloc[0,0]
        expected_x2.append(pred_x2) 
        upper.append(pred_x2 + 1.96*(var_x2**(1/2)))
        lower.append(pred_x2 - 1.96*(var_x2**(1/2)))
    
    # take actual values into list for plotting
    actual_x2 = data2["x2"].to_list()
    actual_x1 = data2["x1"].to_list()
    
    fig, ax = plt.subplots()

    ax.plot([i for i in range(len(actual_x2))], actual_x2, color="blue", label="Actual")
    ax.plot([i for i in range(len(actual_x2))], expected_x2,color = "orange", label="Expected")
    ax.plot([i for i in range(len(actual_x2))], upper, linestyle='dashed', color = "red")
    ax.plot([i for i in range(len(actual_x2))], lower, linestyle='dashed', color = 'red')
    plt.xlabel('ith Observation')
    plt.ylabel('X2 Value')
    plt.legend()

    plt.show()
    
    
    """
    Problem 3
    """
    
    
    # read csv
    data3 = pd.read_csv("problem3.csv")
    
    # examining data
    acf = plot_acf(data3)
    pacf = plot_pacf(data3)
    
    # fit AR model and print summary
    for i in range(1,4):
        print("------------------------------------------")
        print(f"AR({i})")
        # set desired parameters
        model = ARIMA(data3, order=(i, 0, 0))
        model_fitted = model.fit()
        # Print the results
        print(model_fitted.summary())
    
    # fit MA model and print summary
    for i in range(1,4):
        print("------------------------------------------")
        print(f"MA({i})")
        model = ARIMA(data3, order=(0, 0, i))
        model_fitted = model.fit()
        # Print the results
        print(model_fitted.summary())

    