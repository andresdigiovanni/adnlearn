from catboost import CatBoostRegressor


def get_base_model():
    return {'cat_boost_regressor': CatBoostRegressor(verbose=False)}


def get_hyperparameters_model():
    # https://catboost.ai/docs/concepts/python-reference_catboostregressor.html

    param_dist = { }

    clf = CatBoostRegressor()

    model = {'cat_boost_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
