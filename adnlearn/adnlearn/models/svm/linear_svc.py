from sklearn.svm import LinearSVC


def get_base_model():
    return {'linear_svc': LinearSVC()}


def get_hyperparameters_model():
    C = [0.01, 0.1, 1, 3]
    class_weight = ['balanced', None]
    multi_class = ['ovr', 'crammer_singer']

    param_dist = {'cls__C': C,
                  'cls__class_weight': class_weight,
                  'cls__multi_class': multi_class}

    clf = LinearSVC()

    model = {'linear_svc': {'model': clf, 'param_distributions': param_dist}}
    return model
