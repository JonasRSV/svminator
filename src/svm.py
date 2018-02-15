import math
import random as rn
import numpy as np
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def sign(n):
    """K."""
    if n >= 0:
        return 1
    else:
        return -1

def gen_clusters(clusterSpecs, dim):
    totalNumOfPoints = 0
    for spec in clusterSpecs:
        totalNumOfPoints+=spec['number_of_points']
    labels = np.array([0 for _ in range(totalNumOfPoints)])
    matrix = np.zeros((totalNumOfPoints, dim))
    i = 0
    for spec in clusterSpecs:
        for n in range(spec['number_of_points']):
            point = np.array([spec['group_spread'] * (rn.random() - 0.5)
                              for _ in range(dim)])
            matrix[i] = point + spec['center']
            labels[i] = spec['label']
            i+=1
    randomize = np.random.permutation(totalNumOfPoints)
    return (np.matrix(labels[randomize]), matrix[randomize])

        # {label: -1, group_spred: 1, NUMBER_OF_POINTS: 10, center: point}
        
        


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


def simple_kernel(a, b):
    """Linear Kernel Function."""
    return np.dot(np.matrix(a).A1, np.matrix(b).A1)


def poly_kernel(grade):
    """Polynomial Kernel Function."""
    def kernel(a, b):
        return math.pow((np.dot(np.matrix(a).A1, np.matrix(b).A1) + 1), grade)

    return kernel


def radial_basis_kernel(radial_basis):
    """Radial basis kernel Function."""
    def kernel(a, b):
        return math.exp(
            -math.pow(np.dot(a, b), 2)
            / (2 * math.pow(radial_basis, 2)))

    return kernel


class Classifier(object):
    """SVM classifier object."""

    def __init__(self, data, labels, kernel_function=simple_kernel):
        """Constructor."""
        self.kernel_function = kernel_function
        self.data = data
        self.labels = labels

        self.pre_processed_kernel = None
        self.bias = None
        self.support_vectors = None

    def pre_process_classifiers(self):
        """Preprocess first step of cost function."""
        classes = np.array([[x * y for x in self.labels.A1]
                           for y in self.labels.A1])

        kernel_values = np.array([[self.kernel_function(x, y)
                                   for x in self.data]
                                  for y in self.data])

        self.pre_processed_kernel = classes * kernel_values

    def error_function(self, a):
        """Minimize for good SVM."""
        lagrange_mults = np.array([[x * y for x in a] for y in a])
        su = np.sum(lagrange_mults * self.pre_processed_kernel)

        return (su / 2 - np.sum(a))

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

    def learn(self, upperBound=None):
        """Create classifier."""
        self.pre_process_classifiers()

        init_point = np.zeros(len(self.labels.A1))
        bounds = np.array([(0, upperBound) for _ in self.labels.A1])

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
                * self.kernel_function(point, self.data[key, :])

        return classification - self.bias

    def print(self):
        """Plot support vectors."""
        plt.scatter(self.data[:, 0], self.data[:, 1],
                    c=['r' if i == 1 else 'b' for i in self.labels.A1])

        sv_key = rn.choice(list(self.support_vectors.keys()))
        sv = self.data[sv_key, :]

        print("SV indic", self.indicator_func(sv))
        print("Actual Value", self.labels.A1[sv_key])

        xgrid = np.linspace(-5, 5)
        ygrid = np.linspace(-5, 5)

        grid = np.matrix([
            [self.indicator_func(np.matrix([x, y]))
             for y in ygrid]
            for x in xgrid])

        CX = plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
                         colors=('red', 'black', 'blue'),
                         linewidths=(1, 3, 1))

        plt.clabel(CX, fontsize=9, inline=1)
        plt.show()


(l, d) = gen_clusters([{'label': -1,
                        'group_spread': 2,
                        'center': np.array([3, 3]),
                        'number_of_points': 30},
                       {'label': 1,
                        'group_spread': 1,
                        'center': np.array([-4, 1]),
                        'number_of_points': 20}],
                       2)


classifier_linear = Classifier(d, l)
classifier_poly = Classifier(d, l, kernel_function=poly_kernel(2))
classifier_radial = Classifier(d, l,
                               kernel_function=radial_basis_kernel(40))


classifier_linear.learn()
classifier_linear.print()

# classifier_poly.learn()
# classifier_poly.print()

# classifier_radial.print()
# classifier_radial.learn()

