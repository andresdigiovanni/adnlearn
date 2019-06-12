from sklearn.svm import NuSVC


def get_base_model():
    return {'nu_svc': NuSVC(gamma="scale")}


def get_hyperparameters_model():
    nu = [0.1, 0.5, 1]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
    gamma = ['scale', 'auto']

    param_dist = {'cls__nu': nu,
                  'cls__kernel': kernel,
                  'cls__gamma': gamma}

    clf = NuSVC()

    model = {'nu_svc': {'model': clf, 'param_distributions': param_dist}}
    return model
