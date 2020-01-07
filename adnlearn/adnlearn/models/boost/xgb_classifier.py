from xgboost import XGBClassifier


def get_base_model():
    return {'xgb_classifier': XGBClassifier(random_state=7)}


def get_hyperparameters_model():
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=xgbclassifier#xgboost.XGBClassifier

    max_depth = [3, 100]
    n_estimators = [100, 1000]

    param_dist = {'cls__max_depth': max_depth,
                  'cls__n_estimators': n_estimators}

    clf = XGBClassifier(random_state=7)

    model = {'xgb_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
