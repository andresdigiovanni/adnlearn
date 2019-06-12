import numpy as np


def _column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)].tolist()


def x_y_split(dataset, target):
    X = dataset.drop(columns=target)
    y = dataset[target].copy()
    return X, y


def get_features(X, text_columns=[]):
    cols = X._get_numeric_data().columns
    cols = [x for x in cols if x not in text_columns]
    numeric_features = _column_index(X, cols)

    cols = X.select_dtypes(include=['object']).columns
    cols = [x for x in cols if x not in text_columns]
    categorical_features = _column_index(X, cols)

    text_features = _column_index(X, text_columns)

    return numeric_features, categorical_features, text_features
