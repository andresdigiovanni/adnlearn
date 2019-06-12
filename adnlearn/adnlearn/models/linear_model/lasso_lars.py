from sklearn.linear_model import LassoLars


def get_base_model():
    return {'lasso_lars': LassoLars()}


def get_hyperparameters_model():
    param_dist = { }

    clf = LassoLars()

    model = {'lasso_lars': {'model': clf, 'param_distributions': param_dist}}
    return model
