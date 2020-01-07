from sklearn.ensemble import ExtraTreesClassifier


def get_base_model():
    return {'extra_trees_classifier': ExtraTreesClassifier(n_estimators=100)}


def get_hyperparameters_model():
    n_estimators = [100, 1000]
    max_features = ['auto', 'sqrt', 'log2', None]

    param_dist = {'cls__n_estimators': n_estimators,
                  'cls__max_features': max_features}

    clf = ExtraTreesClassifier()

    model = {'extra_trees_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
