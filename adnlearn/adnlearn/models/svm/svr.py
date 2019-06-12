from sklearn.svm import SVR


def get_base_model():
    return {'svr': SVR()}


def get_hyperparameters_model():
    param_dist = { }

    clf = SVR()

    model = {'svr': {'model': clf, 'param_distributions': param_dist}}
    return model
