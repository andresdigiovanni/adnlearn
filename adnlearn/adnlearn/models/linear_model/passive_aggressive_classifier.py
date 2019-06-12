from sklearn.linear_model import PassiveAggressiveClassifier


def get_base_model():
    return {'passive_aggressive_classifier': PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)}


def get_hyperparameters_model():
    C = [0.001, 0.01, 0.1, 1, 10]

    param_dist = {'cls__C': C}

    clf = PassiveAggressiveClassifier()

    model = {'passive_aggressive_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
