from catboost import CatBoostClassifier


def get_base_model():
    return {'cat_boost_classifier': CatBoostClassifier(verbose=False)}


def get_hyperparameters_model():
    # https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html

    n_estimators = [100, None]

    param_dist = {'cls__n_estimators': n_estimators}

    clf = CatBoostClassifier()

    model = {'cat_boost_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
