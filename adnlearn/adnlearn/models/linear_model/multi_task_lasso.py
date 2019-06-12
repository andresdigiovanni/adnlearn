from sklearn.linear_model import MultiTaskLasso


def get_base_model():
    return {'multi_task_lasso': MultiTaskLasso()}


def get_hyperparameters_model():
    param_dist = { }

    clf = MultiTaskLasso()

    model = {'multi_task_lasso': {'model': clf, 'param_distributions': param_dist}}
    return model

