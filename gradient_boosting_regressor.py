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

spec = [
    ('STEPS', nb.uint32),
    ('DEPTH', nb.uint8),
    # ('structure')
]

@nb.jitclass()
class DecisionTreeRegressor:

    def __init__(self, tree_depth):
        self.STEPS = 25
        self.DEPTH = tree_depth
        self.structure = {}

    def fit(self, X, Y):
        # Give each datapoint an ID
        X = np.c_[np.arange(len(X)), X]
        self.recurse_split(X, Y, depth=1)
    
    def recurse_split(self, X_region, Y_region, depth=0):
        

    def get_best_X_split(self, X_region, Y_region):
        best_feat_i = 0
        best_feat_val = 0.0

        best_err = 9999999.9
        
        for feat_i in range(1, X.shape[1]-1):
            feat_min = X_region[:, feat_i].min()
            feat_max = X_region[:, feat_i].max()
            feat_steps = np.linspace(feat_min, feat_max, self.STEPS)

            for feat_step in feat_steps:
                X_Y = np.c_[X_region, Y_region]
                
                XY_left = X_Y[X_region[:, feat_i] < feat_step]
                XY_right = X_Y[X_region[:, feat_i] >= feat_step]

                region_err = get_region_MSE(XY_left[:, -1]) + get_region_MSE(XY_right[:, -1])

                if region_err < best_err:
                    best_err = region_err
                    best_feat_i = feat_i
                    best_feat_val = feat_step
        
        X_Y = np.c_[X_region, Y_region]
        
        best_XY_left = X_Y[X_Y[best_feat_i < best_feat_val]]
        best_XY_right = X_Y[X_Y[best_feat_i >= best_feat_val]]

        X_left, Y_left = best_XY_left[:, :-1], best_XY_left[:, -1]
        X_right, Y_right = best_XY_right[:, :-1], best_XY_right[:, -1]
        
        return X_left, Y_left, X_right, Y_right

    def get_region_MSE(self, Y_region):
        Y_mean = Y_region[:, -1].mean()
        return np.average(np.square(Y - Y_mean))

    
    def predict(self, X):
        pass