from sklearn.neighbors import RadiusNeighborsClassifier


def get_base_model():
    return {'radius_neighbors_classifier': RadiusNeighborsClassifier()}


def get_hyperparameters_model():
    param_dist = {}

    clf = RadiusNeighborsClassifier()

    model = {'radius_neighbors_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
