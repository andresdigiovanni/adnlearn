import pandas as pd
from adnlearn.dataset.split import *
from adnlearn.storage.pickle import load_pickle, save_pickle
from adnlearn.training.model_selection import get_bests_models, get_best_hyperparameter_model
from pandas import set_option
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def set_environment():
    set_option('display.max_columns', 500)
    set_option('display.max_rows', 500)
    set_option('display.width', 1000)
    set_option('precision', 3)


def get_configuration():
    configuration = {}

    # dataset
    configuration["dataset"] = {}
    configuration["dataset"]["model_type"] = "regression"
    configuration["dataset"]["url"] = "dataset/train.csv"
    configuration["dataset"]["test_url"] = "dataset/test.csv"
    configuration["dataset"]["encoding"] = "latin-1"
    configuration["dataset"]["sep"] = ","
    configuration["dataset"]["decimal"] = "."
    configuration["dataset"]["target"] = "medv"

    # summarize data
    configuration["summarize"] = {}
    configuration["summarize"]["show"] = True

    # training
    configuration["training"] = {}
    configuration["training"]["n_best_training_models"] = 2
    configuration["training"]["n_splits_cv"] = 5
    configuration["training"]["scoring"] = "neg_mean_squared_error"

    # test
    configuration["test"] = {}
    configuration["test"]["test_size"] = 0.20
    configuration["test"]["test_predictions"] = 'predictions.csv'

    # model pickle
    configuration["storage"] = {}
    configuration["storage"]["store_model"] = True
    configuration["storage"]["filename"] = 'finalized_model.pkl'

    # statistics
    configuration["statistics"] = {}
    configuration["statistics"]["show"] = True

    return configuration


if __name__ == '__main__':
    
    # 1) Set environment
    set_environment()

    # 2) Configuration
    configuration = get_configuration()

    # 3) Load data
    dataset = pd.read_csv(
        configuration["dataset"]["url"],
        encoding = configuration["dataset"]["encoding"],
        sep = configuration["dataset"]["sep"],
        decimal = configuration["dataset"]["decimal"],
        header = 0)

    print("Loading {} samples with {} attributes".format(len(dataset.index), len(dataset.columns)))

    # 4) Summarize data
    if configuration["summarize"]["show"]:
        print(dataset.head())

    # 5) Prepare Data
    X, y = x_y_split(dataset, configuration["dataset"]["target"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=configuration["test"]["test_size"],
        random_state=7)

    num_features, cat_features, text_features = get_features(X)

    # 6) Evaluate Algorithms
    bests_clf = get_bests_models(
        X_train, y_train,
        num_features, cat_features, text_features,
        configuration["dataset"]["model_type"],
        configuration["training"]["n_best_training_models"],
        configuration["training"]["n_splits_cv"],
        configuration["training"]["scoring"])

    # you can select the models you prefer
    #bests_clf = ["sgd_classifier"]

    # 7) Evaluate Hyper-parameters Algorithms
    model = get_best_hyperparameter_model(
        X_train, y_train,
        num_features, cat_features, text_features,
        bests_clf,
        configuration["training"]["n_splits_cv"],
        configuration["training"]["scoring"])

    # 8) Show bests params
    print("Bests params for %s: %s" % (model['model_name'], model['best_params']))

    # 9) Show model statistics
    if configuration["statistics"]["show"]:
        predicciones = model['model'].predict(X_test)
        print(r2_score(y_test, predicciones))

    # 10) Train best model
    model = model['model'].fit(X, y)

    # 11) Save model
    if configuration["storage"]["store_model"]:
        save_pickle(model, configuration["storage"]["filename"])

    # 12) Load and test model
    if configuration["storage"]["store_model"]:
        model = load_pickle(configuration["storage"]["filename"])

        test_dataset = pd.read_csv(
            configuration["dataset"]["test_url"],
            encoding = configuration["dataset"]["encoding"],
            sep = configuration["dataset"]["sep"],
            decimal = configuration["dataset"]["decimal"],
            header = 0)

        test_dataset["prediction"] = test_dataset.apply(
            lambda s: model.predict(s.values[None])[0], axis=1
        )

        test_dataset.drop(test_dataset.columns.difference(['ID', 'prediction']), axis=1, inplace=True)

        test_dataset.to_csv(configuration["test"]["test_predictions"], index=False)
