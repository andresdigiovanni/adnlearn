from sklearn.neighbors import KNeighborsClassifier


def get_base_model():
    return {'k_neighbors_classifier': KNeighborsClassifier()}


def get_hyperparameters_model():
    n_neighbors = [2, 3, 5, 8]
    weights = ['uniform', 'distance']

    param_dist = {'cls__n_neighbors': n_neighbors,
                  'cls__weights': weights}

    clf = KNeighborsClassifier()

    model = {'k_neighbors_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
