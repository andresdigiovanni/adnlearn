from sklearn.ensemble import AdaBoostClassifier


def get_base_model():
    return {'ada_boost_classifier': AdaBoostClassifier()}


def get_hyperparameters_model():
    n_estimators = [10, 50, 100, 500, 1000]
    learning_rate = [0.01, 0.1, 0.5, 1, 2]

    param_dist = {'cls__n_estimators': n_estimators,
                  'cls__learning_rate': learning_rate}

    clf = AdaBoostClassifier()

    model = {'ada_boost_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
