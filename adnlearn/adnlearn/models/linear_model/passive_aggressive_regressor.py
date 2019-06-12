from sklearn.linear_model import PassiveAggressiveRegressor


def get_base_model():
    return {'passive_aggressive_regressor': PassiveAggressiveRegressor()}


def get_hyperparameters_model():
    param_dist = { }

    clf = PassiveAggressiveRegressor()

    model = {'passive_aggressive_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
