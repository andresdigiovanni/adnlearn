from sklearn.isotonic import IsotonicRegression


def get_base_model():
    return {'isotonic_regression': IsotonicRegression()}


def get_hyperparameters_model():
    param_dist = { }

    clf = IsotonicRegression()

    model = {'isotonic_regression': {'model': clf, 'param_distributions': param_dist}}
    return model
