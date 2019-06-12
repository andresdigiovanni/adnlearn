from lightgbm import LGBMRegressor


def get_base_model():
    return {'light_gbm_regressor': LGBMRegressor()}


def get_hyperparameters_model():
    # https://lightgbm.readthedocs.io/en/latest/Python-Intro.html

    param_dist = { }

    clf = LGBMRegressor()

    model = {'light_gbm_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
