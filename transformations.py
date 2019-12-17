import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class simple_custom_transformation(BaseEstimator, TransformerMixin):

    def __init__(self, feature_names=None, lower=None, upper=None):
        self._feature_names = feature_names
        if isinstance(feature_names, list):
            if not isinstance(lower, list):
                self._lower = [lower for n in feature_names]
            else:
                self._lower = lower
            if not isinstance(upper, list):
                self._upper = [upper for n in feature_names]
            else:
                self._upper = upper
        else:
            self._lower = lower
            self._upper = upper
    
    def fit (self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.copy()
        if isinstance(X, pd.DataFrame):
            for column in self._feature_names:
                X.loc[:, column] = X.loc[:, column].clip(lower=self._lower)
        return X

