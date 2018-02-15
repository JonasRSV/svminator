import random as rn
import numpy as np
from scipy.optimize import minimize
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
        a = np.array([[x*y for x in labels.A1] for y in labels.A1])
        b = np.array([[self.kernel_function(x,y) for x in self.data] for y in self.data])
        self.pre_processed_kernel = a*b

    def error_function(self, a):
        """Minimize for good SVM."""
        aa = np.array([[x*y for x in a] for y in a])
        su = np.sum(aa*self.pre_processed_kernel)
        
        return (su/2 - np.sum(a))
        # a = np.matrix(a)
        # lagrange_multiplies = np.dot(a.T, a)
        # print(lagrange_multiplies)
        # print(a)
        # kernel_values = np.dot(lagrange_multiplies, self.pre_processed_kernel)

        # return np.sum(kernel_values) * 0.5 - np.sum(a)

    def kernel_constrain(self, a):
        """Constraint for the lagrange thingy to work."""
        return np.dot(np.matrix(a).A1, self.labels.A1)

    def determine_bias(self):
        """Determine the bias of the SVM."""
        sv_key = rn.choice(list(self.support_vectors.keys()))
        a_sv = self.data[sv_key, :]

        bias = 0
        for key, alpha in self.support_vectors.items():
            bias += alpha * self.labels.A1[key]\
                * self.kernel_function(a_sv, self.data[key, :])

        self.bias = bias - self.labels.A1[sv_key]
        print("Bias", self.bias)

    def filter_lagrange_multipliers(self, list_of_multipliers):
        """Filter zero or practically zero values of a."""
        self.support_vectors = dict()
        for idx, sv in enumerate(list_of_multipliers):
            if abs(sv) > 0.000001:
                self.support_vectors[idx] = sv
        print("As left", self.support_vectors)

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


        lagrange_multipliers = minimize_ret.x
        self.filter_lagrange_multipliers(lagrange_multipliers)
        self.determine_bias()

    def indicator_func(self, point):
        """Classify a point."""
        classification = 0
        for key, alpha in self.support_vectors.items():
            classification += alpha * self.labels.A1[key]\
                * self.kernel_function(point, self.data[key,:])

        return classification - self.bias

    def print(self):

        plt.scatter(self.data[:,0], self.data[:,1], 
                    c=['r' if i==1 else 'b' for i in self.labels.A1])


        sv_key = rn.choice(list(self.support_vectors.keys()))
        sv = self.data[sv_key, :]
        print("SV indic", self.indicator_func(sv))
        print("Actual Value", self.labels.A1[sv_key])

        xgrid = np.linspace(-25, 25)
        ygrid = np.linspace(-25, 25)

        grid = np.matrix([[self.indicator_func(np.matrix([x,y])) for y in ygrid] for x in xgrid])
        # CX = plt.contour(xgrid, ygrid, grid)
        
        CX = plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
                         colors=('red', 'black', 'blue'),
                         linewidths=(1, 3, 1))

        plt.clabel(CX, fontsize=9, inline=1)
        plt.show()


GROUP_SPREAD = 10
CLUSTER_SEPARATION = 10
NUMBER_OF_POINTS = 400
DIMENSION = 2


(labels, data) = random_clusters(GROUP_SPREAD,
                                 CLUSTER_SEPARATION,
                                 NUMBER_OF_POINTS,
                                 DIMENSION)

classifier = Classifier(data, labels)

bounds = np.array([(0, None) for _ in range(NUMBER_OF_POINTS)])
classifier.learn(bounds)
classifier.print()


