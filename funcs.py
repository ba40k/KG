import numpy as np
import pandas as pd
import sklearn


def entropy(a_list):
    '''
    Calculates and returns entropy of given binary
    1d collection, using
    entropy = -p_0*log2(p_0) - p_1*log2(p_1),
    where
        p_0 - proportion of value 0 in a_list,
        p_1 - proportion of value 1 in a_list,
    '''
    a = np.array(a_list)
    if len(a) == 0:
         return 0 
    p_1 = a.sum() / len(a)
    p_0 = 1 - p_1 
    if a.sum() == 0 or len(a) == a.sum():
        return 0 
    entropy = -p_0 * np.log2(p_0) - p_1 * np.log2(p_1)
    return entropy

def information_gain(root, left, right):
    '''
    Calculates and returns Information Gain 
    if we split root by left and right
    '''
    return entropy(root) - 1/len(root)*(len(left)*entropy(left) + len(right) * entropy(right))


def best_feature_to_split(X, y, to_print=True):
    ''' Выводит прирост информации при разбиении по каждому признаку'''
    """\n Возвращает тройку (имя фичи, прирост информации, массив признаков) """
    best_splits = [] # (fname, ig, val)
    
    for feature in X.columns:
        vals = X[feature].unique()
        pivot_keys = []
        for val in vals:
            mask = (X[feature] == val)
            pivot_key = (y[mask] == 1)
            pivot_keys.append((pivot_key.sum(), val))
        
        pivot_keys = sorted(pivot_keys)
        accum = set()
        best_ig = 0
        left_vals = []
        for pk in pivot_keys:
            accum.add(pk[1])
            mask = (X[feature].isin(accum))
            y_left = y[mask]
            y_right = y[~mask]
            ig = information_gain(y, y_left, y_right)
            if ig > best_ig:
                best_ig = ig
                left_vals = set(accum)
        best_splits.append((feature, best_ig, set(left_vals)))
    
    best_splits = sorted(best_splits, key = lambda x : x[1])
    if to_print:
        print(best_splits)
    
    return best_splits[-1]
