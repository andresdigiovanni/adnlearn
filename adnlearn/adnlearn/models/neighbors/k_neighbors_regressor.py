from sklearn.neighbors import KNeighborsRegressor


def get_base_model():
    return {'k_neighbors_regressor': KNeighborsRegressor()}


def get_hyperparameters_model():
    param_dist = { }

    clf = KNeighborsRegressor()

    model = {'k_neighbors_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
