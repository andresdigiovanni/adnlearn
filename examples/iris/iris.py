import numpy as np
import pandas as pd
from adnlearn.dataset.split import *
from adnlearn.storage.pickle import *
from adnlearn.summarize.model_statistics import plot_confusion_matrix, plot_feature_importances
from adnlearn.summarize.summarize_data import descriptive_statistics, data_visualizations
from adnlearn.training.model_selection import get_bests_models, get_best_hyperparameter_model
from pandas import read_csv, set_option
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def set_environment():
    set_option('display.max_columns', 500)
    set_option('display.max_rows', 500)
    set_option('display.width', 1000)
    set_option('precision', 3)


def get_configuration():
    configuration = {}

    # dataset
    configuration["dataset"] = {}
    configuration["dataset"]["model_type"] = "multi-class"
    configuration["dataset"]["url"] = "dataset/iris.csv"
    configuration["dataset"]["encoding"] = "latin-1"
    configuration["dataset"]["sep"] = ","
    configuration["dataset"]["decimal"] = "."
    configuration["dataset"]["target"] = "class"

    # summarize data
    configuration["summarize"] = {}
    configuration["summarize"]["show"] = False

    # training
    configuration["training"] = {}
    configuration["training"]["n_best_training_models"] = 2
    configuration["training"]["n_splits_cv"] = 5
    configuration["training"]["scoring"] = "accuracy"

    # test
    configuration["test"] = {}
    configuration["test"]["test_size"] = 0.20

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
        descriptive_statistics(dataset, configuration["dataset"]["target"])
        data_visualizations(dataset, configuration["dataset"]["target"])

    # 5) Prepare Data
    X, y = x_y_split(dataset, configuration["dataset"]["target"])
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = configuration["test"]["test_size"],
        random_state = 7,
        stratify = y)

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
    #bests_clf = ["xgb_classifier", "logistic_regression"]

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
        labels = dataset[configuration["dataset"]["target"]].unique().tolist()
        columns = list(X.columns.values)

        plot_feature_importances(model['model']._final_estimator, columns)
        plot_confusion_matrix(X_test, y_test, model['model'], labels)

    # 10) Train best model
    model = model['model'].fit(X, y)

    # 11) Save model
    if configuration["storage"]["store_model"]:
        save_pickle(model, configuration["storage"]["filename"])

    # 12) Load and test model
    if configuration["storage"]["store_model"]:
        model = load_pickle(configuration["storage"]["filename"])
        features = pd.DataFrame(np.array([[5.1, 3.7, 1.5, 0.4]]), columns=["sepal length", "sepal width", "petal length", "petal width"])
        
        result = model.predict(features)
        print("Result of predict test model: %s. Expected value: 0" % (result))
