from sklearn.linear_model import Perceptron


def get_base_model():
    return {'perceptron': Perceptron(max_iter=1000, tol=1e-3)}


def get_hyperparameters_model():
    alpha = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

    param_dist = {'cls__alpha': alpha}

    clf = Perceptron()

    model = {'perceptron': {'model': clf, 'param_distributions': param_dist}}
    return model
