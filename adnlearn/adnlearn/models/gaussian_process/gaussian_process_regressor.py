from sklearn.gaussian_process import GaussianProcessRegressor


def get_base_model():
    return {'gaussian_process_regressor': GaussianProcessRegressor()}


def get_hyperparameters_model():
    param_dist = { }

    clf = GaussianProcessRegressor()

    model = {'gaussian_process_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
