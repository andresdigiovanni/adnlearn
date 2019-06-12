from sklearn.linear_model import RANSACRegressor


def get_base_model():
    return {'ransac_regressor': RANSACRegressor()}


def get_hyperparameters_model():
    param_dist = { }

    clf = RANSACRegressor()

    model = {'ransac_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
