from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def get_base_model():
    return {'quadratic_discriminant_analysis': QuadraticDiscriminantAnalysis()}


def get_hyperparameters_model():
    store_covariance = [True, False]
    tol = [1e-5, 1e-4, 1e-3, 1e-2]

    param_dist = {'cls__store_covariance': store_covariance,
                  'cls__tol': tol}

    clf = QuadraticDiscriminantAnalysis()

    model = {'quadratic_discriminant_analysis': {'model': clf, 'param_distributions': param_dist}}
    return model
