import numpy as np
import numba as nb
from decision_tree_regressor import DecisionTreeRegressor

# TODO choose different loss functions

class GradientBoostingRegressor:

    def __init__(self, iterations=50, tree_depth=3, min_tree_region_datapoints=6):
        self.gradient_estimating_trees = []
        self.iterations = iterations
        self.tree_depth = tree_depth
        self.min_tree_region_datapoints = min_tree_region_datapoints

    def fit(self, X, Y):
        # Usually the first estimator (y_hat) is the mean of all Y's
        Y_mean = Y.mean()
        self.Y_mean_vector = np.array([Y_mean for i in range(len(Y))])

        # Fit regression tree to estimate gradient of loss function (generalizes well on test-data)
        Neg_gradient = self.negative_MSE_gradient(Y, self.Y_mean_vector)

        grad_estimating_tree = DecisionTreeRegressor(self.tree_depth, self.min_tree_region_datapoints)
        grad_estimating_tree.fit(X, Neg_gradient)
        
        self.gradient_estimating_trees.append(grad_estimating_tree)
        
        for i in range(self.iterations-1):
            Prediction = self.predict(X)
            Neg_gradient = self.negative_MSE_gradient(Y, Prediction)
            
            grad_estimating_tree = DecisionTreeRegressor(self.tree_depth, self.min_tree_region_datapoints)
            grad_estimating_tree.fit(X, Neg_gradient)

            self.gradient_estimating_trees.append(grad_estimating_tree)

    
    def predict(self, X):
        Prediction = self.Y_mean_vector

        for reg_tree in self.gradient_estimating_trees:
            Neg_Gradient = reg_tree.predict(X)
            Prediction += Neg_Gradient
        
        return Prediction

    # residual of y and y_hat
    def negative_MSE_gradient(self, Y, Y_hat):
        return -(Y - Y_hat)
