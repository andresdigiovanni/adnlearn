from sklearn.ensemble import ExtraTreesRegressor


def get_base_model():
    return {'extra_trees_regressor': ExtraTreesRegressor(n_estimators=100)}


def get_hyperparameters_model():
    param_dist = { }

    clf = ExtraTreesRegressor()

    model = {'extra_trees_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
