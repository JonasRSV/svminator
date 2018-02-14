import random as rn
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib as mp

PsuperVector = None
LABELS = None
SIZE = 20
CLUSTER_SEP = 100


def sign(n):
    """K."""
    if n >= 0:
        return 1
    else:
        return -1


def random_clusters(dim):
    """Generate two clusters in dim dimensional space."""
    global LABELS
    global CLUSTER_SEP
    global SIZE

    LABELS = [sign(rn.randrange(-10, 10)) for _ in range(SIZE)]

    separator = np.array([CLUSTER_SEP for _ in range(dim)])

    matrix_dim = (SIZE, dim)
    matrix = np.zeros(matrix_dim)
    for i in range(SIZE):
        point = np.array([rn.random() for _ in range(dim)])

        if LABELS[i] == 1:
            point = point + separator
        else:
            point = point - separator

        matrix[i] = point


    dataFrame = pd.DataFrame(matrix)
    dataFrame['label'] = LABELS

    return dataFrame


def simpleKern(a,b):
    return np.dot(a,b)


def preComputePsuper(data, kern):
    global PsuperVector
    labelVector = np.matrix(data['label'])
    PsuperVector = np.matmul(np.transpose(labelVector), (labelVector))

    for x, datax in data.iterrows():
        for y, datay in data.iterrows():
            PsuperVector[x, y] *= kern(datax.drop(['label']), datay.drop(['label']))


def superSum(a):
    aa = np.matmul(a,np.transpose(a))
    ss = aa*PsuperVector
    return np.sum(ss)/2-np.sum(a)



def zerofun(a):
    global LABELS
    return np.dot(a,LABELS) == 0


dataFrame = random_clusters(4)
preComputePsuper(dataFrame, simpleKern)
print(superSum(np.zeros(SIZE)))


aaa = minimize(superSum, np.zeros(SIZE), bounds=np.array([(0, None) for _ in range(SIZE)]), constraints={'type':'eq', 'fun':zerofun})
print(aaa)

