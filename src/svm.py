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


def simpleKern(a, b):
    """Simple kernel."""
    return np.dot(a, b)


def preComputePsuper(data, kern):
    """Simple kernel."""
    global PsuperVector
    labelVector = np.matrix(data['label'])
    PsuperVector = np.matmul(np.transpose(labelVector), (labelVector))

    for x, datax in data.iterrows():
        for y, datay in data.iterrows():
            PsuperVector[x, y] *= kern(datax.drop(['label']), datay.drop(['label']))


def superSum(a):
    """Simple kernel."""
    aa = np.matmul(a, np.transpose(a))
    ss = aa * PsuperVector
    return np.sum(ss) / 2 - np.sum(a)


def zerofun(a):
    """Simple kernel."""
    global LABELS
    return np.dot(a, LABELS) == 0


def getTheFuckingAs(allAs):
    """Simple kernel."""
    b = {}
    for idx, a in enumerate(allAs):
        if abs(a) < 0.00001:
            continue
        b[idx] = a
    return b


def calculateB(allAs, data, kern):
    """Fuck."""
    s = data.iloc(0)
    sum = 0
    for idx, dat in data.iterrows():
        if idx in allAs:
            sum += allAs[idx] * dat['label']\
                * kern(s.drop('label'), dat.drop('label'))

    return sum - s['label']


def indicatorFunc(allAs, b, data, kern, s):
    """Shit."""
    sum = 0
    for idx, dat in data.iterrows():
        if idx in allAs:
            sum += allAs[idx] * dat['label'] * kern(dat.drop('label'), s)
    return sum - b


dataFrame = random_clusters(4)
preComputePsuper(dataFrame, simpleKern)
print(superSum(np.zeros(SIZE)))

print(dataFrame)


aaa = minimize(superSum, np.zeros(SIZE), 
               bounds=np.array([(0, None) for _ in range(SIZE)]),
               constraints={'type': 'eq', 'fun': zerofun})

print(aaa.x)

