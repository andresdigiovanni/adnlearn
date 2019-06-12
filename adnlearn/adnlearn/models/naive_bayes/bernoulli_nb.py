from sklearn.naive_bayes import BernoulliNB


def get_base_model():
    return {'bernoulli_nb': BernoulliNB()}


def get_hyperparameters_model():
    alpha = [0.01, 0.1, 0.5, 1, 2]
    fit_prior = [True, False]

    param_dist = {'cls__alpha': alpha,
                  'cls__fit_prior': fit_prior}

    clf = BernoulliNB()

    model = {'bernoulli_nb': {'model': clf, 'param_distributions': param_dist}}
    return model
