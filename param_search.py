import warnings
warnings.filterwarnings("ignore", category=FutureWarning, append=1)

import pandas
from multiprocessing import Pool
from sklearn import svm
import numpy as np
import sys

CV_SETS = 5
INTERESTING_FEATURES = ('cyls', 'displacement', 'hp',
                        'weight', 'acc', 'year', 'origin')


def get_features_and_labels(csv_file):
    """Read CSV data and extract interesting features and labels"""
    df = pandas.read_csv(csv_file)
    labels = df['mpg'].values
    features = zip(*map(lambda f: df[f].values, INTERESTING_FEATURES))
    return features, labels


def cross_validation_sets(feature_vectors, labels, n=CV_SETS):
    """Construct n sets of features and labels for use in cross-validation."""
    feature_sets = [ [] for i in range(n) ]
    label_sets = [ [] for i in range(n) ]
    feature_it = iter(feature_vectors)
    label_it = iter(labels)
    i = 0
    try:
        while True:
            feature_sets[i].append(feature_it.next())
            label_sets[i].append(label_it.next())
            i = (i + 1) % n
    except:
        pass
    return feature_sets, label_sets


def idx_rotate(idx):
    """Primitive list rotation operation."""
    return idx.insert(0, idx.pop())


def permutation(feature_sets, label_sets, idx):
    """Construct train and test sets for a given index list."""
    train_features, train_labels = [], []
    for i in idx[1:]:
        train_features += feature_sets[i]
        train_labels += label_sets[i]
    return train_features, train_labels, feature_sets[idx[0]], label_sets[idx[0]]


def score_params(feature_sets, label_sets, C, gamma):
    """Run cross-validation on all set permutations for the given SVM params."""
    idx = range(CV_SETS)
    totals = [0] * CV_SETS
    for perm in range(CV_SETS):
        train_f, train_l, test_f, test_l = permutation(feature_sets, label_sets, idx)
        clf = svm.SVR(C=C, gamma=gamma, kernel='rbf', scale_C=False)
        clf.fit(train_f, train_l)
        for i in range(len(test_f)):
            f, l = test_f[i], test_l[i]
            totals[perm] += (l - clf.predict(f)[0]) ** 2
        idx_rotate(idx)

    return C, gamma, sum(totals) / CV_SETS


def test_params(feature_sets, label_sets, C, gamma):
    """Run cross-validation on with the given SVM params using the process pool."""
    return proc_pool.apply(score_params, (feature_sets, label_sets, C, gamma))


def param_permutations(C_min, C_max, C_no, gamma_min, gamma_max, gamma_no):
    """Construct param tuples for within the given window."""
    Cs = np.linspace(C_min, C_max, num=C_no)
    gammas = np.linspace(gamma_min, gamma_max, num=gamma_no)
    out = []
    for c in Cs:
        for gamma in gammas:
            out.append( (c, gamma) )
    return out


def _score_params_map(args):
    """Hack to keep score_params useable, alling it to be used in Pool.map()."""
    return score_params(*args)


def test_param_perms(feature_sets, label_sets, perms):
    """Return the scores for all permutations in perms."""
    aug_perms = map(lambda x: tuple([feature_sets, label_sets] + list(x)), perms)
    return proc_pool.map(_score_params_map, aug_perms)


def hone(C_min, C_max, C_no, gamma_min, gamma_max, gamma_no, feature_sets, label_sets,
         new_window_proportion=0.5, epsilon=0.1, last_error=0, max_iterations=10):
    """Given a starting window and the rate at which the window is 'zoomed',
       continually zoom in on the lowest minimum until there's little change
       or mox_iterations is exceded.  Note: this will quite possibly get stuck
       in a local minimum."""

    # get best (min) params
    perms = param_permutations(C_min, C_max, C_no, gamma_min, gamma_max, gamma_no)
    results = test_param_perms(feature_sets, label_sets, perms)
    C_best, gamma_best, error_best = sorted(results, key=lambda x: x[2])[0]

    print >> sys.stderr
    print >> sys.stderr, "Window: C:", (C_min, C_max), "  gamma:", (gamma_min, gamma_max)
    print >> sys.stderr, "Best C:", C_best, "  gamma:", gamma_best, "  error:", error_best

    # stopping condition(s)
    if (abs(last_error - error_best) <= epsilon) or (max_iterations == 1):
        return C_best, gamma_best, error_best

    # calculate new window size, keeping {C, gamma} > 0
    C_diff = C_max - C_min
    gamma_diff = gamma_max - gamma_min
    C_offset = C_diff * new_window_proportion * 0.5
    gamma_offset = gamma_diff * new_window_proportion * 0.5

    C_min_new = C_best - C_offset
    C_max_new = C_best + C_offset
    if C_min_new < 0:
        C_max_new += abs(C_min_new)
        C_min_new = 1

    gamma_min_new = gamma_best - gamma_offset
    gamma_max_new = gamma_best + gamma_offset
    if gamma_min_new < 0:
        gamma_max_new += abs(gamma_min_new)
        gamma_min_new = 1e-20

    # recursive step
    return hone(C_min_new, C_max_new, C_no, gamma_min_new, gamma_max_new, gamma_no,
                feature_sets, label_sets, new_window_proportion=new_window_proportion,
                epsilon=epsilon, last_error=error_best, max_iterations=max_iterations-1)


def std_dev(Xs):
    """Standard deviation of the items in a list."""
    mean = sum(Xs) / len(Xs)
    return math.sqrt(sum(map(lambda x: (x-mean)**2, Xs))/(len(Xs)-1))

# it appears Pool() needs to be run /after/ the function it'll need to apply
proc_pool = Pool(processes=None)

if __name__ == '__main__':

    # get best params
    feature_sets, label_sets = cross_validation_sets(*get_features_and_labels("auto-mpg-nameless.csv"))
    C, gamma, error = hone(100, 20000, 10, 1e-9, 1e-5, 10, feature_sets, label_sets,
                           new_window_proportion=0.6, max_iterations=3)

    # produce some stats about regression abilities
    import math
    import pylab

    # train on everything
    clf = svm.SVR(kernel='rbf', C=C, gamma=gamma, scale_C=False)
    features = reduce(lambda x,y: x+y, feature_sets)
    labels = reduce(lambda x,y: x+y, label_sets)
    clf.fit(features, labels)

    # calculate errors for all data
    errors = []
    for feature, label in zip(features, labels):
        errors.append(clf.predict(feature)[0] - label)
    errors = sorted(errors)

    print
    print "Standard deviation of error:", std_dev(errors)

    # construct a crude histogram of errors
    granularity = 30
    bins = [0] * granularity
    _min, _max = errors[0], errors[-1]
    diff = _max - _min
    errors = map(lambda x: x-_min, errors)
    step = diff / (granularity - 1)
    bin_nos = map(lambda x: int(math.floor(x/step)), errors)
    for i in bin_nos:
        bins[i] += 1

    # tidy up unused bins
    x, y = [], []
    for a, b in zip(range(granularity), bins):
        if b > 0:
            x.append(_min + a * step)
            y.append(b)

    # plot
    pylab.plot(x, y)
    pylab.show()
