from sklearn.ensemble import ExtraTreesClassifier


def get_base_model():
    return {'extra_trees_classifier': ExtraTreesClassifier(n_estimators=100)}


def get_hyperparameters_model():
    n_estimators = [30, 100, 1000]
    criterion = ['gini', 'entropy']
    max_depth = [20, 100]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    max_features = ['auto', 'sqrt', 'log2', None]
    bootstrap = [True, False]
    class_weight = ['balanced', 'balanced_subsample', None]

    param_dist = {'cls__n_estimators': n_estimators,
                  'cls__criterion': criterion,
                  'cls__max_depth': max_depth,
                  'cls__min_samples_split': min_samples_split,
                  'cls__min_samples_leaf': min_samples_leaf,
                  'cls__max_features': max_features,
                  'cls__bootstrap': bootstrap,
                  'cls__class_weight': class_weight}

    clf = ExtraTreesClassifier()

    model = {'extra_trees_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
