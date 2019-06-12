from sklearn.linear_model import Lars


def get_base_model():
    return {'lars': Lars()}


def get_hyperparameters_model():
    param_dist = { }

    clf = Lars()

    model = {'lars': {'model': clf, 'param_distributions': param_dist}}
    return model
