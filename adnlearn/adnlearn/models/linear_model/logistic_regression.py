from sklearn.linear_model import LogisticRegression


def get_base_model():
    return {'logistic_regression': LogisticRegression(solver="lbfgs", multi_class="auto")}


def get_hyperparameters_model():
    C = [0.01, 0.1, 1, 3]
    class_weight = ['balanced', None]
    solver = ['newton-cg', 'lbfgs', 'sag', 'saga']
    multi_class = ['ovr', 'multinomial', 'auto']

    param_dist = {'cls__C': C,
                  'cls__class_weight': class_weight,
                  'cls__solver': solver,
                  'cls__multi_class': multi_class}

    clf = LogisticRegression()

    model = {'logistic_regression': {'model': clf, 'param_distributions': param_dist}}
    return model
