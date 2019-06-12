from xgboost import XGBClassifier


def get_base_model():
    return {'xgb_classifier': XGBClassifier(random_state=7)}


def get_hyperparameters_model():
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=xgbclassifier#xgboost.XGBClassifier

    max_depth = [3, 10, 100]
    learning_rate = [0.03, 0.1, 1]
    n_estimators = [30, 100, 1000]
    gamma = [0, 0.5, 1]
    min_child_weight = [1, 3]
    subsample = [0.1, 1]
    colsample_bytree = [0.5, 1]

    param_dist = {'cls__max_depth': max_depth,
                  'cls__learning_rate': learning_rate,
                  'cls__n_estimators': n_estimators,
                  'cls__gamma': gamma,
                  'cls__min_child_weight': min_child_weight,
                  'cls__subsample': subsample,
                  'cls__colsample_bytree': colsample_bytree}

    clf = XGBClassifier(random_state=7)

    model = {'xgb_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
