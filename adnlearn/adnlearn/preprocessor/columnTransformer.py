from adnlearn.preprocessor.nlp import process_text
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


# https://www.oreilly.com/library/view/applied-text-analysis/9781491963036/ch04.html
text_transformer = Pipeline(steps=[
    ('bow', CountVectorizer(analyzer=process_text)),
    ('tfidf', TfidfTransformer())])


def get_column_transformer_preprocessor(numeric_features, categorical_features, text_features):
    transformers = []
    
    if len(numeric_features) > 0:
        transformers.append(('num', numeric_transformer, numeric_features))

    if len(categorical_features) > 0:
        transformers.append(('cat', categorical_transformer, categorical_features))

    for x in text_features:
        transformers.append(('txt_' + str(x), text_transformer, x))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor
