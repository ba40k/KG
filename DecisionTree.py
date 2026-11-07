import numpy as np
import pandas as pd
import sklearn
import Estimators
from funcs import *
class DecisionTree:
    """
    Decision tree, that considers binary features and solves
    binary classification task
    """
    class Node:
        """
        Node which are used to build DecisionTree
        """
        def __init__(self, is_leaf=False, val=None, feature=None, depth=None, major = None, estimator_class=Estimators.MajorityEstimator):
            """
            Initalizes a new tree node instance
            
            Parameters
            ----------
            is_leaf, Boolean
                defines whethes it's leaf node (has no children)
            val, int
               то по каким значениям предиктора идет разбиение
            feature, str
                name of feature which is used to split data
                by this node
            depth, int
                depth level of this node
            """
            self.left = None
            self.right = None
            self.is_leaf = is_leaf
            self.val = val
            self.feature = feature
            self.depth = depth
            self.estimator_class = estimator_class
            self.estimator = None
            self.root = None
            
                      
    def __init__(self, max_depth : int =3, estimator_class=Estimators.MajorityEstimator):
        """
        Initalizes a new DecisionTree instance
        """
        self.max_depth=max_depth
        self.root = None
        self.estimator_class = estimator_class
        
    def _make_leaf(self, node, X, y):
        node.is_leaf = True
        node.estimator = self.estimator_class()
        node.estimator.fit(X, y)
        return node

    def build_tree(self, X : pd.DataFrame, y : pd.Series, curr_node : Node=None, curr_depth : int =0) -> Node:
        """
        Returns recursively built classification tree
        """
        if curr_node is None:
            curr_node = self.Node(estimator_class=self.estimator_class)

        if curr_depth == self.max_depth or entropy(list(y)) == 0:
           return self._make_leaf(curr_node, X, y)
        best_split = best_feature_to_split(X, y, to_print=False)
        if best_split[1] <= 0.0001:  
            return self._make_leaf(curr_node, X, y)
        mask = X[best_split[0]].isin(best_split[2])

        X_left = X[mask]
        X_right = X[~mask]
        y_left = y[mask]
        y_right = y[~mask]
        curr_node.val = best_split[2]
        curr_node.depth = curr_depth
        curr_node.feature = best_split[0]

        
        
        curr_node.is_leaf = False 
        curr_node.left = self.build_tree(X_left, y_left,curr_depth=curr_depth+1)
        curr_node.right = self.build_tree(X_right, y_right,curr_depth=curr_depth+1)
                        
        
        return curr_node
        
    def fit(self, X : pd.DataFrame, y : pd.Series):
        """
        Fits to X: builds a decision tree that is
        appropriate to X
        Returns self.
        """
        self.root = self.build_tree(X, y)
        return self

    def predict(self, X : pd.DataFrame, y : pd.Series, cur_node : Node):
        if cur_node is None:
            cur_node = self.root
        if not cur_node.is_leaf:
            fname = cur_node.feature
            left_mask = X[fname].isin(cur_node.val)
            right_mask = ~left_mask

            y.loc[left_mask] = self.predict(
                X.loc[left_mask],
                y.loc[left_mask],
                cur_node.left,
            )

            y.loc[right_mask] = self.predict(
                X.loc[right_mask],
                y.loc[right_mask],
                cur_node.right,
            )
        else :
            y.iloc[:] = cur_node.estimator.predict(X)
        return y