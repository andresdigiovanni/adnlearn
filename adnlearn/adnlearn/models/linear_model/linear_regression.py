from sklearn.linear_model import LinearRegression


def get_base_model():
    return {'linear_regression': LinearRegression()}


def get_hyperparameters_model():
    param_dist = {}

    clf = LinearRegression()

    model = {'linear_regression': {'model': clf, 'param_distributions': param_dist}}
    return model
