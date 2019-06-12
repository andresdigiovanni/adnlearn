from sklearn.neighbors import RadiusNeighborsRegressor 


def get_base_model():
    return {'radius_neighbors_regressor ': RadiusNeighborsRegressor()}


def get_hyperparameters_model():
    param_dist = { }

    clf = RadiusNeighborsRegressor()

    model = {'radius_neighbors_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
