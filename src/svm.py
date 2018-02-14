import random as rn
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib as mp

PsuperVector = None
LABELS = 0
SIZE = 20


def sign(n):
    if n >= 0:
        return 1
    else:
        return -1

def simpleKern(a,b):
    """ A simple kernel. """
    return np.dot(a, b)


def preComputePsuper(data, kern):
    """ A simple kernel. """
    global PsuperVector
    labelVector = np.matrix(data['label'])
    PsuperVector = np.matmul(np.transpose(labelVector), (labelVector))

    for x, datax in data.iterrows():
        for y, datay in data.iterrows():
            PsuperVector[x, y] *= kern(datax.drop(['label']), datay.drop(['label']))

def superSum(a):
    """ A simple kernel. """
    aa = np.matmul(a, np.transpose(a))
    ss = aa * PsuperVector
    return np.sum(ss) / 2 - np.sum(a)

def zerofun(a):
    """ A simple kernel. """
    global LABELS
    return np.dot(a,LABELS) == 0

def getTheFuckingAs(allAs):
    """ A simple kernel. """
    b = {}
    for idx, a in enumerate(allAs):
        if abs(a) < 0.00001:
            continue
        b[ids] = a
    return b

def calculateB(allAs, data, kern):
    s = data.iloc(0)
    sum = 0
    for idx, dat in data.iterrows():
        if idx in allAs:
            sum += allAs[idx]*dat['label']*kern(s.drop('label'), dat.drop('label'))
    return sum - s['label']

def indicatorFunc(allAs, b, data, kern, s):
    sum = 0
    for idx, dat in data.iterrows():
        if idx in allAs:
            sum += allAs[idx]*dat['label']*kern(dat.drop('label'), s)
    return sum - b





dataFrame = pd.DataFrame()
for i in range(3):
    randomArray = [rn.random() for _ in range(SIZE)]
    dataFrame["data " + str(i)] = randomArray

dataFrame["label"] = [sign(rn.randrange(-10, 10)) for _ in range(SIZE)]
LABELS = np.array(dataFrame['label'])
preComputePsuper(dataFrame, simpleKern)
print(superSum(np.zeros(SIZE)))

print(dataFrame)


aaa = minimize(superSum, np.zeros(SIZE), bounds=np.array([(0, None) for _ in range(SIZE)]), constraints={'type':'eq', 'fun':zerofun})

print(aaa.x)

