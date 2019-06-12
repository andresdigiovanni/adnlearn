from sklearn.neural_network import MLPRegressor


def get_base_model():
    return {'mlp_regressor': MLPRegressor()}


def get_hyperparameters_model():
    param_dist = { }

    clf = MLPRegressor()

    model = {'mlp_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
