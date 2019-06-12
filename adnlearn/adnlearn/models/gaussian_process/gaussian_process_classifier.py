from sklearn.gaussian_process import GaussianProcessClassifier


def get_base_model():
    return {'gaussian_process_classifier': GaussianProcessClassifier()}


def get_hyperparameters_model():
    param_dist = { }

    clf = GaussianProcessClassifier()

    model = {'gaussian_process_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
