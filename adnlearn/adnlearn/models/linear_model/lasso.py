from sklearn.linear_model import Lasso


def get_base_model():
    return {'lasso': Lasso()}


def get_hyperparameters_model():
    param_dist = { }

    clf = Lasso()

    model = {'lasso': {'model': clf, 'param_distributions': param_dist}}
    return model

