from scipy import stats
from sklearn.neural_network import MLPClassifier


def get_base_model():
    return {'mlp_classifier': MLPClassifier()}


def get_hyperparameters_model():
    hidden_layer_sizes = [(stats.randint.rvs(100,600,1), stats.randint.rvs(100,600,1),), 
                                          (stats.randint.rvs(100,600,1),)]
    activation = ['identity', 'logistic', 'tanh', 'relu']
    solver = ['lbfgs', 'sgd', 'adam']
    alpha = [0.0001, 0.001, 0.01]
    learning_rate = ['constant', 'invscaling', 'adaptive']

    param_dist = {'cls__hidden_layer_sizes': hidden_layer_sizes,
                  'cls__activation': activation,
                  'cls__solver': solver,
                  'cls__alpha': alpha,
                  'cls__learning_rate': learning_rate}

    clf = MLPClassifier()

    model = {'mlp_classifier': {'model': clf, 'param_distributions': param_dist}}
    return model
