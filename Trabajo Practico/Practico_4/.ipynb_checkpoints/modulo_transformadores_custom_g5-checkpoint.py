from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class MakeDummies(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X1 = pd.get_dummies(X, columns = self.columns, drop_first = False)
        self.feature_names = X1.columns.tolist()
        return X1
    
    def get_feature_names(self):
        return self.feature_names
            
    
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X1 = pd.DataFrame(X[self.features])
        self.feature_names = X1.columns.tolist()
        return pd.DataFrame(X[self.features])
    
    def get_feature_names(self):
        return self.feature_names
      
    
class MakeInt(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X1 = pd.DataFrame(X[self.columns].astype(int))
        self.feature_names = X1.columns.tolist()
        return pd.DataFrame(X[self.columns].astype(int))
    
    def get_feature_names(self):
        return self.feature_names
    