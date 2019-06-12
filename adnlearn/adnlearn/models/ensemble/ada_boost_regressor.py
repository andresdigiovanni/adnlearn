from sklearn.ensemble import AdaBoostRegressor


def get_base_model():
    return {'ada_boost_regressor': AdaBoostRegressor()}


def get_hyperparameters_model():
    param_dist = { }

    clf = AdaBoostRegressor()

    model = {'ada_boost_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
