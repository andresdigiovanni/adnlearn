from sklearn.linear_model import MultiTaskElasticNet


def get_base_model():
    return {'multi_task_elastic_net': MultiTaskElasticNet()}


def get_hyperparameters_model():
    param_dist = { }

    clf = MultiTaskElasticNet()

    model = {'multi_task_elastic_net': {'model': clf, 'param_distributions': param_dist}}
    return model
