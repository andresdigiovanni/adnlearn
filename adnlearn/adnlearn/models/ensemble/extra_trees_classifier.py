from sklearn.ensemble import ExtraTreesClassifier


def get_base_model():
    return {'extra_trees_classifier': ExtraTreesClassifier(n_estimators=100)}


def get_hyperparameters_model():
    n_estimators = [100, 1000]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]

    param_dist = {'cls__n_estimators': n_estimators,
                  'cls__min_samples_split': min_samples_split,
                  'cls__min_samples_leaf': min_samples_leaf}

    clf = ExtraTreesClassifier()

    model = {'extra_trees_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
