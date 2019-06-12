from sklearn.svm import LinearSVR


def get_base_model():
    return {'linear_svr': LinearSVR()}


def get_hyperparameters_model():
    param_dist = { }

    clf = LinearSVR()

    model = {'linear_svr': {'model': clf, 'param_distributions': param_dist}}
    return model
