import random as rn
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt

PsuperVector = None
LABELS = None
SIZE = 40
CLUSTER_SEP = 2
GROUP_SPREAD = 2

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
    global GROUP_SPREAD

    LABELS = np.array([sign(rn.randrange(-10, 10)) for _ in range(SIZE)])
    separator = np.array([CLUSTER_SEP for _ in range(dim)])

    matrix_dim = (SIZE, dim)
    matrix = np.zeros(matrix_dim)

    for i in range(SIZE):
        point = np.array([GROUP_SPREAD * (rn.random()-0.5) for _ in range(dim)])
        matrix[i] = point + LABELS[i] * separator

    dataFrame = pd.DataFrame(matrix)
    return dataFrame


def simpleKern(a, b):
    """Simple kernel."""
    return np.dot(a, b)


def preComputePsuper(data, kern):
    """Simple kernel."""
    global PsuperVector
    global LABELS
    
    PsuperVector = np.matmul(np.transpose(np.matrix(LABELS)), np.matrix(LABELS))
    for x, datax in data.iterrows():
        for y, datay in data.iterrows():
            PsuperVector[x, y] *= kern(datax, datay)


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
    s = np.array(data.ix[1])
    su = 0
    for idx, dat in data.iterrows():
        if idx in allAs:
            su += allAs[idx] * LABELS[idx]\
                * kern(s, dat)

    return su - LABELS[0]


def indicatorFunc(allAs, b, data, kern, s):
    """Shit."""
    su = 0
    for idx, dat in data.iterrows():
        if idx in allAs:
            su += allAs[idx] * LABELS[idx] * kern(dat, s)
    return su - b


dataFrame = random_clusters(2)
preComputePsuper(dataFrame, simpleKern)

print(superSum(np.zeros(SIZE)))
print(dataFrame)

aaa = minimize(superSum, np.zeros(SIZE), 
               bounds=np.array([(0, None) for _ in range(SIZE)]),
               constraints={'type': 'eq', 'fun': zerofun})

print(aaa)

aas = getTheFuckingAs(aaa.x)

b = calculateB(aas, dataFrame, simpleKern)

print(b)

print(indicatorFunc(aas, b, dataFrame, simpleKern, np.array([100, 100])))
print(indicatorFunc(aas, b, dataFrame, simpleKern, np.array([-100, -100])))

dfw = dataFrame.copy()
dfw['label'] = LABELS
dp = dfw[dfw['label']==1]
dp.columns = ['A', 'B', 'L']
plt.scatter(dp['A'], dp['B'])

dp = dfw[dfw['label']==-1]
dp.columns = ['A', 'B', 'L']
plt.scatter(dp['A'], dp['B'])

xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-5, 5)

grid = np.array([[indicatorFunc(aas, b, dataFrame, simpleKern, np.array([x,y])) for y in ygrid] for x in xgrid])

plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidth=(1,3,1))

plt.show()
