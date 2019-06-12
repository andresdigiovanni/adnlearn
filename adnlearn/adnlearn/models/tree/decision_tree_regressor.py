from sklearn.tree import DecisionTreeRegressor


def get_base_model():
    return {'decision_tree_regressor': DecisionTreeRegressor()}


def get_hyperparameters_model():
    param_dist = { }

    clf = DecisionTreeRegressor()

    model = {'decision_tree_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
