from sklearn.ensemble import GradientBoostingRegressor


def get_base_model():
    return {'gradient_boosting_regressor': GradientBoostingRegressor()}


def get_hyperparameters_model():
    param_dist = { }

    clf = GradientBoostingRegressor()

    model = {'gradient_boosting_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
