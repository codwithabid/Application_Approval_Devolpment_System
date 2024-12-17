import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class KNNImputerCustom(BaseEstimator, TransformerMixin):
    def __init__(self, columns, n_neighbors=5):
        self.columns = columns
        self.knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X[self.columns] = self.knn_imputer.fit_transform(X[self.columns])
        return X
    
# Custom transformer for OneHotEncoding
class OneHotEncoderCustom(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols
        self.ohe = OneHotEncoder(drop='first', sparse_output=False)  # Changed sparse to sparse_output
    
    def fit(self, X, y=None):
        self.ohe.fit(X[self.categorical_cols])
        return self
    
    def transform(self, X):
        encoded_cols = pd.DataFrame(self.ohe.transform(X[self.categorical_cols]), columns=self.ohe.get_feature_names_out(self.categorical_cols))
        X = X.drop(columns=self.categorical_cols)
        X = pd.concat([X, encoded_cols], axis=1)
        return X
    
# Custom transformer for Outlier Imputation
class OutlierImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method='median'):
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)].index
            if self.method == 'median':
                X.loc[outliers, col] = X[col].median()
            elif self.method == 'mean':
                X.loc[outliers, col] = X[col].mean()
        return X

class StandardScalerWithExclusion(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_column='LoanApproved'):
        """
        Initialize the StandardScalerWithExclusion transformer.
        This will apply StandardScaler to all numeric columns except for the column to be excluded.
        """
        self.exclude_column = exclude_column
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        columns_to_scale = X.select_dtypes(include=['number']).drop(columns=[self.exclude_column], errors='ignore').columns
        self.scaler.fit(X[columns_to_scale])
        
        return self

    def transform(self, X):
        columns_to_scale = X.select_dtypes(include=['number']).drop(columns=[self.exclude_column], errors='ignore').columns
        X[columns_to_scale] = self.scaler.transform(X[columns_to_scale])
        
        return X
