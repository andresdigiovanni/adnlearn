from sklearn.kernel_ridge import KernelRidge


def get_base_model():
    return {'kernel_ridge': KernelRidge()}


def get_hyperparameters_model():
    param_dist = { }

    clf = KernelRidge()

    model = {'kernel_ridge': {'model': clf, 'param_distributions': param_dist}}
    return model
