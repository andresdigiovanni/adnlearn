from sklearn.ensemble import GradientBoostingClassifier


def get_base_model():
    return {'gradient_boosting_classifier': GradientBoostingClassifier()}


def get_hyperparameters_model():
    learning_rate = [0.01, 0.1, 0.5]
    n_estimators = [10, 50, 100, 500, 1000]
    subsample = [0.01, 0.1, 1, 2]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    max_depth = [2, 3, 5, 9]
    max_features = ['auto', 'sqrt', 'log2', None]

    param_dist = {'cls__learning_rate': learning_rate,
                  'cls__n_estimators': n_estimators,
                  'cls__subsample': subsample,
                  'cls__min_samples_split': min_samples_split,
                  'cls__min_samples_leaf': min_samples_leaf,
                  'cls__max_depth': max_depth,
                  'cls__max_features': max_features}

    clf = GradientBoostingClassifier()

    model = {'gradient_boosting_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
