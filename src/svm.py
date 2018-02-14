import random as rn
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mp

PsuperVector = None

def sign(n):
    if n >= 0:
        return 1
    else:
        return -1

def simpleKern(a,b):
    return np.dot(a,b)


def preComputePsuper(data, kern):
    labelVector = np.array(data['label'])
    PsuperVector = np.matmul(np.transpose(labelVector), (labelVector))

    for x, datax in data.iterrows():
        for y, datay in data.iterrows():
            PsuperVector[x][y] *= kern(datax.drop(['label']), datay.drop(['label']))


def superSum(a):
    aa = np.matmul(a,np.transpose(a))
    ss = aa*PsuperVector
    return np.sum(ss)/2-np.sum(a)




dataFrame = pd.DataFrame()
for i in range(3):
    randomArray = [rn.random() for _ in range(20)]
    dataFrame["data " + str(i)] = randomArray

dataFrame["label"] = [sign(rn.randrange(-10, 10)) for _ in range(20)]

preComputePsuper(dataFrame, simpleKern)
print(superSum(np.zeros(20)))


