from scipy import stats
from sklearn.semi_supervised import LabelPropagation


def get_base_model():
    return {'label_propagation': LabelPropagation()}


def get_hyperparameters_model():
    kernel = ['knn', 'rbf']
    gamma = stats.uniform(10, 40)
    n_neighbors = stats.randint(3, 20)

    param_dist = {'cls__kernel': kernel,
                  'cls__gamma': gamma,
                  'cls__n_neighbors': n_neighbors}

    clf = LabelPropagation()

    model = {'label_propagation': {'model': clf, 'param_distributions': param_dist}}
    return model
