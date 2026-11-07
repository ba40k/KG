import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LeafEstimator:
    def fit(self, X : pd.DataFrame, y : pd.Series):
        pass 

    def predict(self, X : pd.DataFrame) -> pd.Series:
        pass 

class MajorityEstimator(LeafEstimator):
    def __init__(self):
        super().__init__()
        self.majority = None
       
    def fit(self, X, y):
        self.majority = y.mean().round()        

    def predict(self, X):
        return pd.Series(self.majority, index=X.index)

class KnnEstimator(LeafEstimator):
    def __init__(self, n_neighbors = 3):
            super().__init__()
            self.knn = None
            self.n_neighbors= n_neighbors
    def fit(self, X, y):
        k = min(len(y), self.n_neighbors)
        self.knn =sklearn.neighbors.KNeighborsClassifier(n_neighbors = k)        
        self.knn.fit(X, y)
    def predict(self, X):
        return pd.Series(self.knn.predict(X), index=X.index)



class LogRegEstimator(LeafEstimator):
    def __init__(self):
        self.model = LogisticRegression()
        self.constant_class = None

    def fit(self, X, y):
        self.constant_class = None

        if y.nunique() == 1:
            self.constant_class = y.iloc[0]
        else:
            self.model.fit(X, y)

        return self

    def predict(self, X):
        if self.constant_class is not None:
            return pd.Series(self.constant_class, index=X.index)

        return pd.Series(
            self.model.predict(X),
            index=X.index,
        )


class LDAEstimator(LeafEstimator):
    def __init__(self):
        self.model = LinearDiscriminantAnalysis()
        self.constant_class = None

    def fit(self, X, y):
        self.constant_class = None

        if y.nunique() == 1:
            self.constant_class = y.iloc[0]
        else:
            self.model.fit(X, y)

        return self

    def predict(self, X):
        if self.constant_class is not None:
            return pd.Series(self.constant_class, index=X.index)

        return pd.Series(
            self.model.predict(X),
            index=X.index,
        )
