from xgboost import XGBRegressor


def get_base_model():
    return {'xgb_regressor': XGBRegressor(random_state=7)}


def get_hyperparameters_model():
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=xgbclassifier#xgboost.XGBRegressor

    param_dist = { }

    clf = XGBRegressor(random_state=7)

    model = {'xgb_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
