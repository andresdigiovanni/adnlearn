from catboost import CatBoostClassifier


def get_base_model():
    return {'cat_boost_classifier': CatBoostClassifier(verbose=False)}


def get_hyperparameters_model():
    # https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html

    n_estimators = [50, 100, 500, 1000, 2000]
    learning_rate = [0.01, 0.03, 0.1, 0.5, 1]
    depth = [4, 6, 8, 16]
    l2_leaf_reg = [2, 3, 5, 7]

    param_dist = {'cls__n_estimators': n_estimators,
                  'cls__learning_rate': learning_rate,
                  'cls__depth': depth,
                  'cls__l2_leaf_reg': l2_leaf_reg}

    clf = CatBoostClassifier()

    model = {'cat_boost_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
