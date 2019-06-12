from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

def train_classifiers(X_train, y_train, models, n_splits_cv, scoring):
    score_models = {}
    
    for model_name, model in list(models.items()):
        try:
            kfold = KFold(
                n_splits = n_splits_cv,
                random_state = 7)
            cv_results = cross_val_score(
                model,
                X_train, y_train,
                cv = kfold,
                scoring = scoring,
                n_jobs = -1)
            score_models[model_name] = {'score': cv_results.mean()}

            msg = "%s: %f" % (model_name, cv_results.mean())
            print(msg)

        except Exception as e:
            print('An exception occurred with ' + model_name + '. Exception: ' + str(e))

    return score_models


def train_hyperparameter_classifiers(X_train, y_train, models, n_splits_cv, scoring):
    dict_models = {}

    for model_name, model in list(models.items()):
        try:
            clf = GridSearchCV(
                estimator = model['model'],
                param_grid = model['param_distributions'],
                cv = n_splits_cv,
                scoring = scoring,
                n_jobs = -1)

            clf.fit(X_train, y_train)
            best_model = clf.best_estimator_

            dict_models[model_name] = {
                'model_name': model_name,
                'model': best_model,
                'score': clf.best_score_,
                'best_params': clf.best_params_}

            msg = "%s: %f" % (model_name, clf.best_score_)
            print(msg)

        except Exception as e:
            print('An exception occurred with ' + model_name + '. Exception: ' + str(e))

    return dict_models
