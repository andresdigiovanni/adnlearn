from sklearn.linear_model import SGDClassifier


def get_base_model():
    return {'sgd_classifier': SGDClassifier(max_iter=1000, tol=1e-3)}


def get_hyperparameters_model():
    loss = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
    alpha = [0.0001, 0.001, 0.01, 0.1, 1]

    param_dist = {'cls__loss': loss,
                  'cls__alpha': alpha}

    clf = SGDClassifier(max_iter=1000, tol=1e-3)

    model = {'sgd_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
