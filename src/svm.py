import random as rn
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib as mp

PsuperVector = None
LABELS = 0

def sign(n):
    if n >= 0:
        return 1
    else:
        return -1

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


SIZE = 20

dataFrame = pd.DataFrame()
for i in range(3):
    randomArray = [rn.random() for _ in range(SIZE)]
    dataFrame["data " + str(i)] = randomArray

dataFrame["label"] = [sign(rn.randrange(-10, 10)) for _ in range(SIZE)]
LABELS = np.array(dataFrame['label'])
preComputePsuper(dataFrame, simpleKern)
print(superSum(np.zeros(SIZE)))


aaa = minimize(superSum, np.zeros(SIZE), bounds=np.array([(0, None) for _ in range(SIZE)]), constraints={'type':'eq', 'fun':zerofun})
print(aaa)

