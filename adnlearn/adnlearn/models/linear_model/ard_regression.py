from sklearn.linear_model import ARDRegression


def get_base_model():
    return {'ard_regression': ARDRegression()}


def get_hyperparameters_model():
    param_dist = { }

    clf = ARDRegression()

    model = {'ard_regression': {'model': clf, 'param_distributions': param_dist}}
    return model
