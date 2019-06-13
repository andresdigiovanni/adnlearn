import numpy as np
import pandas as pd
from adnlearn.models.selector import *
from adnlearn.preprocessor.columnTransformer import get_column_transformer_preprocessor
from adnlearn.training.model_trainer import *
from sklearn.pipeline import Pipeline


# select

def _get_bests_scored_models(models, num_models):
    models = sorted(models.items(), key=lambda k_v: k_v[1]['score'], reverse=True)
    models = [i[0] for i in models]

    return models[:num_models]


def _get_better_scored_hyperparameter_model(models):
    models = sorted(models.items(), key=lambda k_v: k_v[1]['score'], reverse=True)
    return models[0][1]


# pipelines

def _get_pipelines(models, preprocessor):
    pipelines = {}
    for model_name, model in list(models.items()):
        pipelines[model_name] = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('cls', model)])

    return pipelines


def _get_pipelines_hyperparameter(models, preprocessor):
    pipelines = {}
    for model_name, model in list(models.items()):
        pipelines[model_name] = {'model': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('cls', model['model'])]),
                                 'param_distributions': model['param_distributions']}

    return pipelines


# print

def _print_result(models, sort_by='score'):
    cls = [key for key in models.keys()]
    score = [models[key]['score'] for key in cls]
    time = [models[key]['time'] for key in cls]
    
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),3)), columns = ['model_name', 'score', 'time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'model_name'] = cls[ii]
        df_.loc[ii, 'score'] = score[ii]
        df_.loc[ii, 'time'] = time[ii]
    
    print(df_.sort_values(by=sort_by, ascending=False))


# public methods

def get_bests_models(
    X_train, y_train,
    num_features, cat_features, text_features,
    model_type,
    n_best_training_models, n_splits_cv, scoring):

    print("Evaluating models...")

    models = get_models(model_type)
    preprocessor = get_column_transformer_preprocessor(num_features, cat_features, text_features)
    pipelines = _get_pipelines(models, preprocessor)

    score_models = train_classifiers(X_train, y_train, pipelines, n_splits_cv, scoring)
    _print_result(score_models)
    
    bests_models = _get_bests_scored_models(score_models, n_best_training_models)
    return bests_models


def get_best_hyperparameter_model(
    X_train, y_train,
    num_features, cat_features, text_features,
    models,
    n_splits_cv, scoring):

    print("Evaluating hyper-parameters models...")

    models = get_hyperparameter_models(models)
    preprocessor = get_column_transformer_preprocessor(num_features, cat_features, text_features)
    pipelines = _get_pipelines_hyperparameter(models, preprocessor)

    score_models = train_hyperparameter_classifiers(X_train, y_train, pipelines, n_splits_cv, scoring)
    _print_result(score_models)
    
    better_model = _get_better_scored_hyperparameter_model(score_models)
    return better_model
