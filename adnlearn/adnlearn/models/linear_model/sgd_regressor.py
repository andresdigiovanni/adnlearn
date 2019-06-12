from sklearn.linear_model import SGDRegressor


def get_base_model():
    return {'sgd_regressor': SGDRegressor()}


def get_hyperparameters_model():
    param_dist = { }

    clf = SGDRegressor()

    model = {'sgd_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
