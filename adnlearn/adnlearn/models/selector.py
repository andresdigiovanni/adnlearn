from adnlearn.models.boost import *
from adnlearn.models.discriminant_analysis import *
from adnlearn.models.ensemble import *
from adnlearn.models.gaussian_process import *
from adnlearn.models.isotonic import *
from adnlearn.models.kernel_ridge import *
from adnlearn.models.linear_model import *
from adnlearn.models.naive_bayes import *
from adnlearn.models.neighbors import *
from adnlearn.models.neural_network import *
from adnlearn.models.semi_supervised import *
from adnlearn.models.svm import *
from adnlearn.models.tree import *


def _get_multi_classification_models():
    # https://scikit-learn.org/stable/modules/multiclass.html

    models = {}

    # Linear Models
    models.update(sgd_classifier.get_base_model())
    models.update(perceptron.get_base_model())
    models.update(logistic_regression.get_base_model())
    models.update(passive_aggressive_classifier.get_base_model())
    models.update(ridge_classifier.get_base_model())
    # Discriminant Analysis
    models.update(linear_discriminant_analysis.get_base_model())
    models.update(quadratic_discriminant_analysis.get_base_model())
    # Support vector machines
    models.update(linear_svc.get_base_model())
    models.update(svc.get_base_model())
    models.update(nu_svc.get_base_model())
    # Nearest Neighbors
    models.update(k_neighbors_classifier.get_base_model())
    models.update(nearest_centroid.get_base_model())
    models.update(radius_neighbors_classifier.get_base_model())
    # Gaussian Processes
    models.update(gaussian_process_classifier.get_base_model())
    # Naive Bayes
    models.update(gaussian_nb.get_base_model())
    models.update(multinomial_nb.get_base_model())
    models.update(complement_nb.get_base_model())
    models.update(bernoulli_nb.get_base_model())
    # Decision Trees
    models.update(decision_tree_classifier.get_base_model())
    models.update(extra_tree_classifier.get_base_model())
    # Semi supervised
    models.update(label_propagation.get_base_model())
    models.update(label_spreading.get_base_model())
    # Ensemble
    models.update(ada_boost_classifier.get_base_model())
    models.update(random_forest_classifier.get_base_model())
    models.update(extra_trees_classifier.get_base_model())
    models.update(gradient_boosting_classifier.get_base_model())
    # Boost
    models.update(xgb_classifier.get_base_model())
    models.update(light_gbm_classifier.get_base_model())
    models.update(cat_boost_classifier.get_base_model())
    # Neural Network
    models.update(mlp_classifier.get_base_model())

    return models


def _get_regression_models():
    # https://scikit-learn.org/stable/supervised_learning.html

    models = {}

    # Linear Models
    models.update(linear_regression.get_base_model())
    models.update(ridge_regression.get_base_model())
    models.update(lasso.get_base_model())
    models.update(multi_task_lasso.get_base_model())
    models.update(elastic_net.get_base_model())
    models.update(multi_task_elastic_net.get_base_model())
    models.update(lars.get_base_model())
    models.update(lasso_lars.get_base_model())
    models.update(orthogonal_matching_pursuit.get_base_model())
    models.update(bayesian_ridge.get_base_model())
    models.update(ard_regression.get_base_model())
    models.update(logistic_regression.get_base_model())
    models.update(sgd_classifier.get_base_model())
    models.update(sgd_regressor.get_base_model())
    models.update(perceptron.get_base_model())
    models.update(passive_aggressive_classifier.get_base_model())
    models.update(passive_aggressive_regressor.get_base_model())
    models.update(ransac_regressor.get_base_model())
    models.update(theil_sen_regressor.get_base_model())
    models.update(huber_regressor.get_base_model())
    # Kernel ridge regression
    models.update(kernel_ridge.get_base_model())
    ## Support vector machines
    models.update(linear_svr.get_base_model())
    models.update(svr.get_base_model())
    models.update(nu_svr.get_base_model())
    # Nearest Neighbors
    models.update(k_neighbors_regressor.get_base_model())
    models.update(radius_neighbors_regressor.get_base_model())
    # Gaussian Processes
    models.update(gaussian_process_regressor.get_base_model())
    # Decision Trees
    models.update(decision_tree_regressor.get_base_model())
    models.update(extra_tree_regressor.get_base_model())
    ## Semi supervised
    models.update(label_propagation.get_base_model())
    models.update(label_spreading.get_base_model())
    # Ensemble
    models.update(ada_boost_regressor.get_base_model())
    models.update(random_forest_regressor.get_base_model())
    models.update(extra_trees_regressor.get_base_model())
    models.update(gradient_boosting_regressor.get_base_model())
    # Isotonic regression
    models.update(isotonic_regression.get_base_model())
    # Boost
    models.update(xgb_regressor.get_base_model())
    models.update(light_gbm_regressor.get_base_model())
    models.update(cat_boost_regressor.get_base_model())
    # Neural Network
    models.update(mlp_regressor.get_base_model())

    return models


def get_models(model_type):
    models = {}
    
    if model_type == "multi-class":
        models = _get_multi_classification_models()

    elif model_type == "regression":
        models = _get_regression_models()
    
    return models


def get_hyperparameter_models(req_models):
    models = {}

    for model_name in req_models:
        models.update(getattr(globals()[model_name], 'get_hyperparameters_model')())

    return models
