from sklearn.ensemble import RandomForestRegressor

    
def get_base_model():
    return {'random_forest_regressor': RandomForestRegressor(random_state=7, n_estimators=100)}


def get_hyperparameters_model():
    param_dist = { }

    clf = RandomForestRegressor(random_state=7)

    model = {'random_forest_regressor': {'model': clf, 'param_distributions': param_dist}}
    return model
