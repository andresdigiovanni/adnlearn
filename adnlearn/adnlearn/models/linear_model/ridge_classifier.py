from sklearn.linear_model import RidgeClassifier


def get_base_model():
    return {'ridge_classifier': RidgeClassifier()}


def get_hyperparameters_model():
    alpha = [0, 0.0001, 0.001, 0.01, 0.1, 1]
    solver = {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'}

    param_dist = {'cls__alpha': alpha,
                  'cls__solver': solver}

    clf = RidgeClassifier()

    model = {'ridge_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
