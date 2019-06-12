from sklearn.linear_model import HuberRegressor


def get_base_model():
    return {'huber_regressor': HuberRegressor()}


def get_hyperparameters_model():
    param_dist = { }

    clf = HuberRegressor()

    model = {'huber_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
