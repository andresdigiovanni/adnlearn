from sklearn.semi_supervised import LabelSpreading


def get_base_model():
    return {'label_spreading': LabelSpreading()}


def get_hyperparameters_model():
    kernel = ['knn', 'rbf']
    n_neighbors = [3, 7, 11]
    alpha = [0.2, 0.8]

    param_dist = {'cls__kernel': kernel,
                  'cls__n_neighbors': n_neighbors,
                  'cls__alpha': alpha}

    clf = LabelSpreading()

    model = {'label_spreading': {'model': clf, 'param_distributions': param_dist}}
    return model
