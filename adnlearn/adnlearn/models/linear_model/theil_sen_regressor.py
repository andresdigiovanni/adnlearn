from sklearn.linear_model import TheilSenRegressor


def get_base_model():
    return {'theil_sen_regressor': TheilSenRegressor()}


def get_hyperparameters_model():
    param_dist = { }

    clf = TheilSenRegressor()

    model = {'theil_sen_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
