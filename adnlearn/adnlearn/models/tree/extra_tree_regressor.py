from sklearn.tree import ExtraTreeRegressor


def get_base_model():
    return {'extra_tree_regressor': ExtraTreeRegressor()}


def get_hyperparameters_model():
    param_dist = { }

    clf = ExtraTreeRegressor()

    model = {'extra_tree_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
