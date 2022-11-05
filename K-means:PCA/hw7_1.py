#!/usr/bin/env python3

import numpy as np
import random
import matplotlib.pyplot as plt


def init():
    """
    Initialization Function
    """
    random.seed(20211208)
    # To reproduce the same result,
    # we intentionally set the seed value of random.seed()
    # If you want to get radom numbers,
    # call random.seed() without any argument

def g_center(g):
    g = np.array(g)
    return g.mean(axis=0)
def distance(x,y):
    return sum((x-y)**2)/len(x)
def kmeans(data, k, seed=None, niter=30):
    logs = []
    if seed is None:
        Data = data.tolist()
        seed = sorted(random.sample(Data, k))
    assert k == len(seed), "Need k seed numbers"

    centers = data[np.random.choice(len(data), size=k, replace=False)]

    for t in range(niter):
        group = {}
        for i in range(k):
            group[i] =  []
            #find nearest center
        for row in data:
            temp = []
            for i in range(k):
                temp.append(distance(seed[i],row))
            group[np.argmin(temp)].append(row.tolist())
        #plot data store
        for i in range(k):
            g_temp = np.array(group[i])
            g_temp = np.c_[g_temp, np.full(len(g_temp), i)]
            if i == 0:
                clusters = g_temp
            else:
                clusters = np.append(clusters, g_temp, axis = 0)
        #update center
        centers_n = []
        for i in range(k):
            centers_n.append(g_center(group[i]).tolist())
        centers_n = np.array(centers_n)
        if np.sum(centers - centers_n) == 0:
            break
        else:
            seed = centers_n
            logs.append(clusters)
    return sorted(clusters, key=lambda x: x[2]), seed


if __name__ == "__main__":
    init()
    # Load data
    data = np.load("k_means.data.npy")
    clusters, centers = kmeans(data, 3)
    
    for i in range(len(clusters)):
        x=clusters[i][0]
        y=clusters[i][1]
        if clusters[i][2] == 0:
            plt.scatter(x, y, marker='o', c='y')
        elif clusters[i][2] == 1:
            plt.scatter(x, y, marker='o', c='r')
        elif clusters[i][2] == 2:
            plt.scatter(x, y, marker='o', c='b')
    for i in range(len(centers)):
        a=centers[i][0]
        b=centers[i][1]
        plt.scatter(a, b, marker = '^', c = 'black')
    plt.show()