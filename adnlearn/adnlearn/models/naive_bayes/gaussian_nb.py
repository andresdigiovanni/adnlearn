from sklearn.naive_bayes import GaussianNB


def get_base_model():
    return {'gaussian_nb': GaussianNB()}


def get_hyperparameters_model():
    var_smoothing = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

    param_dist = {'cls__var_smoothing': var_smoothing}

    clf = GaussianNB()

    model = {'gaussian_nb': {'model': clf, 'param_distributions': param_dist}}
    return model
