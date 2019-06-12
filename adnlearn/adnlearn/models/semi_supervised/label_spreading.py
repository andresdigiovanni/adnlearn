from scipy import stats
from sklearn.semi_supervised import LabelSpreading


def get_base_model():
    return {'label_spreading': LabelSpreading()}


def get_hyperparameters_model():
    kernel = ['knn', 'rbf']
    gamma = stats.uniform(10, 40)
    n_neighbors = stats.randint(3, 20)
    alpha = stats.uniform(0.01, 0.6)

    param_dist = {'cls__kernel': kernel,
                  'cls__gamma': gamma,
                  'cls__n_neighbors': n_neighbors,
                  'cls__alpha': alpha}

    clf = LabelSpreading()

    model = {'label_spreading': {'model': clf, 'param_distributions': param_dist}}
    return model
