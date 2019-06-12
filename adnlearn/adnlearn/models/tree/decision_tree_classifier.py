from sklearn.tree import DecisionTreeClassifier


def get_base_model():
    return {'decision_tree_classifier': DecisionTreeClassifier()}


def get_hyperparameters_model():
    criterion = ['gini', 'entropy']
    splitter = ['best', 'random']
    max_depth = [10, 20, 50, 100]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    max_features = ['auto', 'sqrt', 'log2', None]
    class_weight = ['balanced', None]

    param_dist = {'cls__criterion': criterion,
                  'cls__splitter': splitter,
                  'cls__max_depth': max_depth,
                  'cls__min_samples_split': min_samples_split,
                  'cls__min_samples_leaf': min_samples_leaf,
                  'cls__max_features': max_features,
                  'cls__class_weight': class_weight}

    clf = DecisionTreeClassifier()

    model = {'decision_tree_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
