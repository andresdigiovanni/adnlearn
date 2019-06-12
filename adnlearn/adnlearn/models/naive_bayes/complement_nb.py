from sklearn.naive_bayes import ComplementNB


def get_base_model():
    return {'complement_nb': ComplementNB()}


def get_hyperparameters_model():
    alpha = [0, 0.1, 0.5, 1, 2]
    fit_prior = [True, False]
    norm = [True, False]

    param_dist = {'cls__alpha': alpha,
                  'cls__fit_prior': fit_prior,
                  'cls__norm': norm}

    clf = ComplementNB()

    model = {'complement_nb': {'model': clf, 'param_distributions': param_dist}}
    return model
