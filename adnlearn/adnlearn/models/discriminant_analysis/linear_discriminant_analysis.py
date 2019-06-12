from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_base_model():
    return {'linear_discriminant_analysis': LinearDiscriminantAnalysis()}


def get_hyperparameters_model():
    solver = ['svd', 'lsqr', 'eigen']
    shrinkage = ['auto', None]

    param_dist = {'cls__solver': solver,
                  'cls__shrinkage': shrinkage}

    clf = LinearDiscriminantAnalysis()

    model = {'linear_discriminant_analysis': {'model': clf, 'param_distributions': param_dist}}
    return model
