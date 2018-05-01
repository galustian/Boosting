import numpy as np
import numba as nb
from decision_tree_regressor import DecisionTreeRegressor

class GradientBoostingRegressor:

    def __init__(self, iterations=50, tree_depth=3, min_tree_region_datapoints=6):
        self.iterations = iterations
        self.tree_depth = tree_depth
        self.min_tree_region_datapoints = min_tree_region_datapoints

    def fit(self, X, Y):
        pass
    
    def predict(X):
        pass

    def MSE_loss():
        pass

    def MSE_gradient():
        pass
