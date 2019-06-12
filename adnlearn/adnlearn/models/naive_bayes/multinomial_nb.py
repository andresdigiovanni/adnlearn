from sklearn.naive_bayes import MultinomialNB


def get_base_model():
    return {'multinomial_nb': MultinomialNB()}


def get_hyperparameters_model():
    alpha = [0.1, 1.0, 10.0, 100.0]
    fit_prior = [True, False]

    param_dist = {'cls__alpha': alpha,
                  'cls__fit_prior': fit_prior}

    clf = MultinomialNB()

    model = {'multinomial_nb': {'model': clf, 'param_distributions': param_dist}}
    return model
