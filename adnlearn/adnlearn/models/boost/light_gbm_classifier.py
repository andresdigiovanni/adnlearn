from lightgbm import LGBMClassifier


def get_base_model():
    return {'light_gbm_classifier': LGBMClassifier()}


def get_hyperparameters_model():
    # https://lightgbm.readthedocs.io/en/latest/Python-Intro.html

    learning_rate = [0.001, 0.01, 0.03, 0.1]
    n_estimators = [30, 100, 1000, 2000]
    num_leaves = [10, 31, 50]

    param_dist = {'cls__learning_rate': learning_rate,
                  'cls__n_estimators': n_estimators,
                  'cls__num_leaves': num_leaves}

    clf = LGBMClassifier()

    model = {'light_gbm_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
