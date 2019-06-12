from sklearn.svm import SVC


def get_base_model():
    return {'svc': SVC(gamma="auto")}


def get_hyperparameters_model():
    C = [1, 10, 100, 1000]
    kernel = ['linear','poly', 'rbf', 'sigmoid', 'precomputed']
    gamma = [1, 0.1, 0.001, 0.0001, 'auto']

    param_dist = {'cls__C': C,
                  'cls__kernel': kernel,
                  'cls__gamma': gamma}

    clf = SVC()

    model = {'svc': {'model': clf, 'param_distributions': param_dist}}
    return model
