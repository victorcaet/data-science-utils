import pandas as pd
import numpy as np

import itertools

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression

from transformations import simple_custom_transformation

def create_dataset():
    X = pd.DataFrame(data={
        'Categoria': ['a', 'b', 'e', 'c', 'd', 'a', 'b', 'a', 'c', 'd', 'a', 'e', 'a', 'c', 'd', 'a', 'b', 'a', 'c', 'd', 'a', 'b', 'a', 'c', 'd', 'a', 'b', 'a', 'c', 'd', 'a', 'b', 'a', 'c', 'd', 'a', 'b', 'a', 'c', 'd', 'a', 'b', 'a', 'c', 'd'],
        'Dia da semana': ['segunda', 'terça', 'quarta', 'quinta', 'sexta', 'segunda', 'terça', 'quarta', 'quinta', 'sexta', 'segunda', 'terça', 'quarta', 'quinta', 'sexta', 'segunda', 'terça', 'quarta', 'quinta', 'sexta', 'segunda', 'terça', 'quarta', 'quinta', 'sexta', 'segunda', 'terça', 'quarta', 'quinta', 'sexta', 'segunda', 'terça', 'quarta', 'quinta', 'sexta', 'segunda', 'terça', 'quarta', 'quinta', 'sexta', 'segunda', 'terça', 'quarta', 'quinta', 'sexta'],
        'Temperatura Ambiente': np.random.uniform(low=24, high=32, size=(45)).tolist(),
        'Produção': np.random.uniform(low=13.7, high=130.9, size=(45)).tolist()
    })
    X['y'] = np.nan
    coef_c = {'a': 2.4, 'b': 5.1, 'c': 4.13, 'd': 7, 'e': 0.42}
    coef_s = {'segunda': -3.14, 'terça': -6.23, 'quarta': -1.0512, 'quinta': -2.051, 'sexta': 3192e-5}

    for c, s in itertools.product(coef_c.keys(), coef_s.keys()):
        aux = X.loc[((X['Categoria'] == c)&(X['Dia da semana'] == s))].copy()
        X.loc[aux.index, 'y'] = 13 + coef_c[c] * aux['Produção'] - np.exp(coef_s[s]) * aux['Temperatura Ambiente'] + np.random.normal(size=aux.shape[0]) 
        
    X.loc[[0,2,9,7,13,27,33], 'Temperatura Ambiente'] = np.nan
    X.loc[[3,5,9,7,12,22,23], 'Produção'] = np.nan
    return X


def main():
    X = create_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X.drop(['y'], axis=1), X['y'], test_size=0.2, random_state=9)

    # Pipeline for the numeric features including custom transformation
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('clipper', simple_custom_transformation(feature_names=['Temperatura Ambiente', 'Produção'], lower=[26.7, 30], upper=None))
    ])

    # Categorical variables pipeline
    cat_transformer = Pipeline(steps=[
        ('one-hot encoder', OneHotEncoder())
    ])

    # Column transformer to preprocess numeric and categorical data in a different way
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, ['Temperatura Ambiente', 'Produção']),
        ('cat', cat_transformer, ['Categoria', 'Dia da semana'])
    ])

    # Create the pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('lm', LinearRegression())
    ])

    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    results = pd.DataFrame(data={'y_real': y_test.values, 'y_hat': yhat.reshape(-1)}, index=y_test.index)
    print(results)

if __name__ == '__main__':
    main()