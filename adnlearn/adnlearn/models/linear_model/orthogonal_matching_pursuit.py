from sklearn.linear_model import OrthogonalMatchingPursuit


def get_base_model():
    return {'orthogonal_matching_pursuit': OrthogonalMatchingPursuit()}


def get_hyperparameters_model():
    param_dist = { }

    clf = OrthogonalMatchingPursuit()

    model = {'orthogonal_matching_pursuit': {'model': clf, 'param_distributions': param_dist}}
    return model
