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

    dataFrame = pd.DataFrame(matrix)
    return labels, dataFrame


class Classifier(object):
    """SVM classifier object."""

    def __init__(self, kernel_function=None):
        """Constructor."""
        self.kernel_function = kernel_function
        self.data = None
        self.labels = None
        self.pre_processed_kernel = None
        self.bias = None
        self.support_vectors = None

    def pre_process_classifiers(self, data, labels):
        """Preprocess first step of cost function."""
        self.data = data
        self.labels = labels

        matrix_labels = np.matrix(self.labels)

        self.pre_processed_kernel =\
            np.matmul(np.transpose(matrix_labels), matrix_labels)

        for idx, datax in self.data.iterrows():
            for idy, datay in self.data.iterrows():
                self.pre_processed_kernel[idx, idy] *=\
                    self.kernel_function(datax, datay)

    def error_function(self, a):
        """Minimize for good SVM."""
        lagrange_multiplies = np.matmul(a, np.transpose(a))
        kernel_values = lagrange_multiplies * self.pre_processed_kernel

        return 0.5 * np.sum(kernel_values) - np.sum(a)

    def kernel_constrain(self, a):
        """Constraint for the lagrange thingy to work."""
        return np.dot(a, self.labels) == 0

    def filter_lagrange_multipliers(self, list_of_multipliers):
        """Filter zero or practically zero values of a."""
        self.support_vectors = dict()
        for idx, sv in enumerate(list_of_multipliers):
            if abs(sv) > 0.00001:
                self.support_vectors[idx] = sv

    def determine_bias(self):
        """Determine the bias of the SVM."""
        sv_key = rn.choice(self.support_vectors.keys())
        a_sv = self.data[sv_key]

        bias = 0
        for index, point in self.data.iterrows():
            if index in self.support_vectors:
                bias += self.support_vectors[index]\
                    * self.kernel_function(a_sv, point)

        self.bias = bias - self.labels[sv_key]

    def learn(self, data, labels, bounds):
        """Create classifier."""
        self.pre_process_classifiers(data, labels)

        point_cardinality = len(labels)

        lagrange_multipliers =\
            minimize(self.error_function,
                     point_cardinality,
                     bounds=bounds,
                     constraints={'type': 'eq',
                                  'fun': self.kernel_constrain}).x

        self.filter_lagrange_multipliers(lagrange_multipliers)
        self.determine_bias()

    def indicator_func(self, point):
        """Classify a point."""
        classification = 0
        for key, lagrange_a in self.support_vectors.items():
            classification += lagrange_a * self.labels[key]\
                * self.kernel_function(point, self.data[key])\

        return classification - self.bias


def simpleKern(a, b):
    """Linear Kernel Function."""
    return np.dot(a, b)


GROUP_SPREAD = 10
CLUSTER_SEPARATION = 10
NUMBER_OF_POINTS = 20
DIMENSION = 3


(labels, dataFrame) = random_clusters(GROUP_SPREAD,
                                      CLUSTER_SEPARATION,
                                      NUMBER_OF_POINTS,
                                      DIMENSION)

bounds = np.array([(0, None) for _ in range(NUMBER_OF_POINTS)])

classifier = Classifier(kernel_function=simpleKern)

print(dataFrame)

classifier.learn(dataFrame, labels, bounds)


# print(indicatorFunc(aas, b, dataFrame, simpleKern, np.array([100, 100])))
# print(indicatorFunc(aas, b, dataFrame, simpleKern, np.array([-100, -100])))

# dfw = dataFrame.copy()
# dfw['label'] = LABELS
# dp = dfw[dfw['label']==1]
# dp.columns = ['A', 'B', 'L']
# plt.scatter(dp['A'], dp['B'])

# dp = dfw[dfw['label']==-1]
# dp.columns = ['A', 'B', 'L']
# plt.scatter(dp['A'], dp['B'])

# xgrid = np.linspace(-5, 5)
# ygrid = np.linspace(-5, 5)

# grid = np.array([[indicatorFunc(aas, b, dataFrame, simpleKern, np.array([x,y])) for y in ygrid] for x in xgrid])

# plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidth=(1,3,1))

# plt.show()
