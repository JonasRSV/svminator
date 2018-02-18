import math
import random as rn
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def sign(n):
    """K."""
    if n >= 0:
        return 1
    else:
        return -1


def gen_clusters(cluster_specs, dim):
    """Generate clusters."""
    total_number_of_points = 0
    for spec in cluster_specs:
        total_number_of_points += spec['number_of_points']
    labels = np.array([0 for _ in range(total_number_of_points)])
    matrix = np.zeros((total_number_of_points, dim))
    i = 0
    for spec in cluster_specs:
        for n in range(spec['number_of_points']):
            point = np.array([spec['group_spread'] * (rn.random() - 0.5)
                              for _ in range(dim)])
            matrix[i] = point + spec['center']
            labels[i] = spec['label']
            i += 1
    randomize = np.random.permutation(total_number_of_points)
    return (np.matrix(labels[randomize]), matrix[randomize])


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
        delta = np.matrix(a - b).A1

        return math.exp(-np.dot(delta, delta)
                        / (2 * math.pow(radial_basis, 2)))

    return kernel


class Classifier(object):
    """SVM classifier object."""

    def __init__(self, data, labels, kernel_function=simple_kernel, slack=None):
        """Constructor."""
        self.kernel_function = kernel_function
        self.data = data
        self.labels = labels
        self.slack = slack

        self.pre_processed_kernel = None
        self.bias = None
        self.support_weigths = None

    def pre_process_classifiers(self):
        """Preprocess first step of cost function."""
        classes = np.array([[x * y for x in self.labels.A1]
                           for y in self.labels.A1])

        kernel_values = np.array([[self.kernel_function(x, y)
                                   for x in self.data]
                                  for y in self.data])

        self.pre_processed_kernel = classes * kernel_values

    def error_function(self, alphas):
        """Minimize for good SVM."""
        lagrange_mults_matrix = np.array([[x * y for x in alphas]
                                          for y in alphas])

        su = np.sum(lagrange_mults_matrix * self.pre_processed_kernel)

        return (su / 2 - np.sum(alphas))

    def kernel_constrain(self, alphas):
        """Constraint for the lagrange thingy to work."""
        return np.dot(np.matrix(alphas).A1, self.labels.A1)

    def determine_bias(self):
        """Determine the bias of the SVM."""
        sv_key = rn.choice(list(self.support_weigths.keys()))
        a_sv = self.data[sv_key, :]

        bias = 0
        for key, alpha in self.support_weigths.items():
            bias += alpha * self.labels.A1[key]\
                * self.kernel_function(a_sv, self.data[key, :])

        self.bias = bias - self.labels.A1[sv_key]
        print("Bias", self.bias)

    def filter_lagrange_multipliers(self, list_of_multipliers):
        """Filter zero or practically zero values of a."""
        self.support_weigths = dict()
        for idx, sv in enumerate(list_of_multipliers):
            if abs(sv) > 0.000001:
                self.support_weigths[idx] = sv

    def learn(self):
        """Create classifier."""
        self.pre_process_classifiers()

        init_point = np.zeros(len(self.labels.A1))
        bounds = np.array([(0, self.slack) for _ in init_point])

        minimize_ret =\
            minimize(self.error_function,
                     init_point,
                     bounds=bounds,
                     constraints={'type': 'eq',
                                  'fun': self.kernel_constrain})

        alphas = minimize_ret.x
        self.filter_lagrange_multipliers(alphas)
        self.determine_bias()

    def indicator_func(self, point):
        """Classify a point."""
        classification = 0
        for key, alpha in self.support_weigths.items():
            classification += alpha * self.labels.A1[key]\
                * self.kernel_function(point, self.data[key, :])

        return classification - self.bias

    def print(self, plotter):
        """Plot support vectors."""
        # print("errors:", self.errors)
        print("alphas:", self.support_weigths)

        z_axis = []
        for point in self.data:
            z_axis.append(self.indicator_func(point))

        plotter.scatter(xs=self.data[:, 1], ys=self.data[:, 0], zs=z_axis,
                    c=['r' if i == 1 else 'b' for i in self.labels.A1])

        # sv_key = rn.choice(list(self.support_vectors.keys()))
        # sv = self.data[sv_key, :]

        xgrid = np.linspace(-5, 5)
        ygrid = np.linspace(-5, 5)

        grid = np.matrix([
            [self.indicator_func(np.matrix([x, y]))
             for y in ygrid]
            for x in xgrid])

        CX = plotter.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0),
                         colors=('red', 'black', 'blue'),
                         linewidths=(1, 3, 1))

        plotter.clabel(CX, fontsize=9, inline=1)


(l, d) = gen_clusters([{'label': -1,
                        'group_spread': 2,
                        'center': np.array([3, 3]),
                        'number_of_points': 20},
                       {'label': 1,
                        'group_spread': 2,
                        'center': np.array([3, -3]),
                        'number_of_points': 40},
                        {'label': -1,
                        'group_spread': 2,
                        'center': np.array([-3, -3]),
                        'number_of_points': 40},
                        {'label': 1,
                        'group_spread': 2,
                        'center': np.array([-3, 3]),
                        'number_of_points': 60}
                       ],
                      2)


d[20] = np.array([1.5, 1.5])
l[:, 20] = -1

classifier_linear = Classifier(d, l, slack=1)
classifier_poly = Classifier(d, l,
                             kernel_function=poly_kernel(2), slack=0.0202)
classifier_radial = Classifier(d, l,
                               kernel_function=radial_basis_kernel(1))


# f, axarr = plt.subplots(3, sharex=True, sharey=True)
# f.suptitle('Sharing both axes')

ax = Axes3D(plt.gcf())

# classifier_linear.learn()
# classifier_linear.print(axarr[0])

classifier_poly.learn()
classifier_poly.print(ax)

# classifier_radial.learn()
# classifier_radial.print(ax)

# f.subplots_adjust(hspace=0.2)
plt.show()

