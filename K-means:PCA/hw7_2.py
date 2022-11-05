#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

def pca(data, k):
    '''
    Step 1.
    Implement PCA algorithm
    return k PCs from the given data
    '''
    norm_data=(data-np.average(data,axis=0))/data.std(axis=0)
    norm_data_mean= norm_data.mean(axis=0)
    cov_norm_data=(norm_data-norm_data_mean).T.dot((norm_data-norm_data_mean))/(norm_data.shape[0]-1)
    
    e_val, e_vec = np.linalg.eig(cov_norm_data)
    # Make a list of (eigenvalue, eigenvector) tuples
    e_pairs = [(np.abs(e_val[i]), e_vec[:,i]) for i in range(len(e_val))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    e_pairs.sort(key=lambda x: x[0], reverse=True)
    res= e_pairs[0:2]
    return res

def transform_w_pca(data, pcs):
    
    '''
    Step 2.
    Express the original data with PCs.
    '''
    w = np.hstack((pcs[0][1].reshape(7,1),pcs[1][1].reshape(7,1)))
    x=(data-np.average(data,axis=0))/data.std(axis=0)
    converted_data=x.dot(w)
    return converted_data
    

if __name__ == '__main__':
    data = np.loadtxt('HR_comma_sep.csv', delimiter=',', skiprows=1, usecols=range(8))
    data=np.delete(data,6,axis=1)
    pcs = pca(data, 2)
    converted_data = transform_w_pca(data, pcs)

    # Step 3.
    # Plot transformed data with matplotlib.
    # You can plot the data with a 2-dimentional graph.    
    plt.scatter(converted_data.T[0], converted_data.T[1],color='r')
    plt.show()