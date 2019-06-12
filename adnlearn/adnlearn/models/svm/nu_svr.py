from sklearn.svm import NuSVR


def get_base_model():
    return {'nu_svr': NuSVR()}


def get_hyperparameters_model():
    param_dist = { }

    clf = NuSVR()

    model = {'nu_svr': {'model': clf, 'param_distributions': param_dist}}
    return model
