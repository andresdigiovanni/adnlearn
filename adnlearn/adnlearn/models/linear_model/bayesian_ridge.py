from sklearn.linear_model import BayesianRidge


def get_base_model():
    return {'bayesian_ridge': BayesianRidge()}


def get_hyperparameters_model():
    param_dist = { }

    clf = BayesianRidge()

    model = {'bayesian_ridge': {'model': clf, 'param_distributions': param_dist}}
    return model
