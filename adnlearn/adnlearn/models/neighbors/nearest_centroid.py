from sklearn.neighbors import NearestCentroid


def get_base_model():
    return {'nearest_centroid': NearestCentroid()}


def get_hyperparameters_model():
    metric = ['euclidean', 'manhattan']

    param_dist = {'cls__metric': metric}

    clf = NearestCentroid()

    model = {'nearest_centroid': {'model': clf, 'param_distributions': param_dist}}
    return model
