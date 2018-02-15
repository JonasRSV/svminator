import random as rn
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt


def sign(n):
    """K."""
    if n >= 0:
        return 1
    else:
        return -1


def random_clusters(group_spread, cluster_separation, number_of_points, dims):
    """Generate two clusters in dim dimensional space."""
    labels = np.array([sign(rn.randrange(-10, 10))
                      for _ in range(number_of_points)])

    separator = np.array([cluster_separation for _ in range(dims)])

    matrix_dim = (number_of_points, dims)
    matrix = np.zeros(matrix_dim)

    for i in range(number_of_points):
        point = np.array([group_spread * (rn.random() - 0.5)
                         for _ in range(dims)])

        matrix[i] = point + labels[i] * separator

    return np.matrix(labels), matrix

def simpleKern(a, b):
    """Linear Kernel Function."""
    return np.dot(np.matrix(a).A1, np.matrix(b).A1)


class Classifier(object):
    """SVM classifier object."""

    def __init__(self, data, lables, kernel_function=simpleKern):
        """Constructor."""
        self.kernel_function = kernel_function
        self.data = data
        self.labels = lables

        self.pre_processed_kernel = None
        self.bias = None
        self.support_vectors = None

    def pre_process_classifiers(self):
        """Preprocess first step of cost function."""

        self.pre_processed_kernel = np.matmul(labels.T, labels)

        for idx, rowx in enumerate(self.data):
            for idy, rowy in enumerate(self.data):
                self.pre_processed_kernel[idx, idy] *=\
                    self.kernel_function(rowx, rowy)
        print(self.pre_processed_kernel)

    def error_function(self, a):
        """Minimize for good SVM."""
        a = np.matrix(a)
        lagrange_multiplies = np.matmul(a.T, a)
        kernel_values = lagrange_multiplies * self.pre_processed_kernel

        tmp = np.sum(kernel_values) - 2 * np.sum(a)
        return tmp

    def kernel_constrain(self, a, p=False):
        """Constraint for the lagrange thingy to work."""
        return np.dot(np.matrix(a).A1, self.labels.A1)

    def determine_bias(self):
        """Determine the bias of the SVM."""
        sv_key = rn.choice(list(self.support_vectors.keys()))
        a_sv = np.matrix(self.data[sv_key,:])

        bias = 0
        for index, point in enumerate(self.data):
            if index in self.support_vectors:
                bias += self.support_vectors[index]\
                    * self.kernel_function(a_sv, point)

        self.bias = bias - self.labels.A1[sv_key]

    def filter_lagrange_multipliers(self, list_of_multipliers):
        """Filter zero or practically zero values of a."""
        self.support_vectors = dict()
        for idx, sv in enumerate(list_of_multipliers):
            if abs(sv) > 0.00001:
                self.support_vectors[idx] = sv

    def learn(self, bounds):
        """Create classifier."""
        self.pre_process_classifiers()

        init_point = np.zeros(len(labels.A1))

        minimize_ret =\
            minimize(self.error_function,
                     init_point,
                     bounds=bounds,
                     constraints={'type': 'eq',
                                  'fun': self.kernel_constrain})
        print(minimize_ret)
        lagrange_multipliers = minimize_ret.x
        self.filter_lagrange_multipliers(lagrange_multipliers)
        self.determine_bias()

    def indicator_func(self, point):
        """Classify a point."""
        classification = 0
        for key, lagrange_a in self.support_vectors.items():
            classification += lagrange_a * self.labels.A1[key]\
                * self.kernel_function(point, self.data[key,:])

        return classification - self.bias

    def print(self):

        plt.scatter(self.data[:,0], self.data[:,1], c=['r' if i==1 else 'b' for i in self.labels.A1])

        xgrid = np.linspace(-15, 15)
        ygrid = np.linspace(-15, 15)

        print("Bias", self.bias)

    
        grid = np.matrix([[self.indicator_func(np.matrix([x,y])) for y in ygrid] for x in xgrid])
        
        plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
                    colors=('red', 'black', 'blue'),
                    linewidths=(1, 3, 1))
        plt.show()


GROUP_SPREAD = 10
CLUSTER_SEPARATION = 10
NUMBER_OF_POINTS = 250
DIMENSION = 2


(labels, data) = random_clusters(GROUP_SPREAD,
                                 CLUSTER_SEPARATION,
                                 NUMBER_OF_POINTS,
                                 DIMENSION)

classifier = Classifier(data, labels)

bounds = np.array([(0, None) for _ in range(NUMBER_OF_POINTS)])
classifier.learn(bounds)
classifier.print()


# print(indicatorFunc(aas, b, dataFrame, simpleKern, np.array([100, 100])))
# print(indicatorFunc(aas, b, dataFrame, simpleKern, np.array([-100, -100])))


