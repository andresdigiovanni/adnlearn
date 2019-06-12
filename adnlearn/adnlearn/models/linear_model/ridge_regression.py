from sklearn.linear_model import Ridge


def get_base_model():
    return {'ridge_regression': Ridge()}


def get_hyperparameters_model():
    param_dist = { }

    clf = Ridge()

    model = {'ridge_regression': {'model': clf, 'param_distributions': param_dist}}
    return model

