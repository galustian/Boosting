import numpy as np
import numba as nb

class GradientBoostingRegressor:

    def __init__(self, iterations=50, tree_depth=3):
        self.iterations = iterations
        self.tree_depth = tree_depth

    def fit(self, X, Y):
        pass
    
    def predict(X):
        pass

    def MSE_loss():
        pass

    def MSE_gradient():
        pass

class DecisionTreeRegressor:

    def __init__(self, tree_depth):
        self.tree_depth = tree_depth

    def fit(X, Y):
        pass
    
    def recurse_split(self, X_region, Y_region, depth):
        pass
    
    def predict(X):
        pass