import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


def _pretty_time(time):
    minutes, seconds = divmod(time, 60)
    return "{:0>2}:{:05.2f}".format(int(minutes), seconds)


def train_models(X_train, y_train, models, n_splits_cv, scoring):
    score_models = {}
    
    for model_name, model in list(models.items()):
        try:
            kfold = KFold(
                n_splits = n_splits_cv,
                random_state = 7)

            start = time.time()
            cv_results = cross_val_score(
                model,
                X_train, y_train,
                cv = kfold,
                scoring = scoring,
                n_jobs = -1)
            total_time = _pretty_time(time.time() - start)

            score_models[model_name] = {'score': cv_results.mean(), 'time' : total_time}

            msg = "%s: %f (%s)" % (model_name, cv_results.mean(), total_time)
            print(msg)

        except Exception as e:
            print('An exception occurred with ' + model_name + '. Exception: ' + str(e))

    return score_models


def train_hyperparameter_models(X_train, y_train, models, n_splits_cv, scoring):
    dict_models = {}

    for model_name, model in list(models.items()):
        try:
            clf = GridSearchCV(
                estimator = model['model'],
                param_grid = model['param_distributions'],
                cv = n_splits_cv,
                scoring = scoring,
                n_jobs = -1)

            start = time.time()
            clf.fit(X_train, y_train)
            total_time = _pretty_time(time.time() - start)

            best_model = clf.best_estimator_

            dict_models[model_name] = {
                'model_name': model_name,
                'model': best_model,
                'score': clf.best_score_,
                'time': total_time,
                'best_params': clf.best_params_}

            msg = "%s: %f (%s)" % (model_name, clf.best_score_, total_time)
            print(msg)

        except Exception as e:
            print('An exception occurred with ' + model_name + '. Exception: ' + str(e))

    return dict_models
