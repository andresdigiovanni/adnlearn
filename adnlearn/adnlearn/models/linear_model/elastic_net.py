from sklearn.linear_model import ElasticNet


def get_base_model():
    return {'elastic_net': ElasticNet()}


def get_hyperparameters_model():
    param_dist = { }

    clf = ElasticNet()

    model = {'elastic_net': {'model': clf, 'param_distributions': param_dist}}
    return model

