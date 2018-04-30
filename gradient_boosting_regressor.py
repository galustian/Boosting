import numpy as np
import numba as nb

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

spec = [
    ('STEPS', nb.uint32),
    ('DEPTH', nb.uint8),
    ('MIN_DATAPOINTS', nb.uint16),
    # ('structure')
]

# @nb.jitclass(spec)
class DecisionTreeRegressor:

    def __init__(self, tree_depth=3, min_datapoints=6):
        self.STEPS = 25
        self.DEPTH = tree_depth
        self.MIN_DATAPOINTS = min_datapoints
        self.structure = {}

    def fit(self, X, Y):
        # Give datapoints an ID
        X = np.c_[np.arange(len(X)), X]
        self.recurse_split(X, Y, structure=self.structure, depth=1)
    
    def recurse_split(self, X_region, Y_region, structure=None, depth=0):
        if 'END_NODE' in structure: 
            return

        split = get_best_X_split(X_region, Y_region)

        X_left, Y_left = split['X_left'], split['Y_left']
        X_right, Y_right = split['X_right'], split['Y_right']
        feat_i, feat_val = split['feat_i'], split['feat_val']
        
        # TODO maybe variable importance?
        # best_err = split['best_err']

        structure['feat_i'] = feat_i
        structure['feat_val'] = feat_val

        # If maximum Depth reached: construct right- and left endnodes
        if depth == self.DEPTH:
            structure['left_next'] = {}
            structure['left_next']['END_NODE'] = True
            structure['left_next']['prediction'] = Y_left.mean()
            
            structure['right_next'] = {}
            structure['right_next']['END_NODE'] = True
            structure['right_next']['prediction'] = Y_right.mean()
            return

        # If left of right less than min. datapoints, construct respective endnode
        if len(X_left) < self.MIN_DATAPOINTS:
            structure['left_next'] = {}
            structure['left_next']['END_NODE'] = True
            structure['left_next']['prediction'] = Y_left.mean()
        if len(X_right) < self.MIN_DATAPOINTS:
            structure['right_next'] = {}
            structure['right_next']['END_NODE'] = True
            structure['right_next']['prediction'] = Y_right.mean()

        if 'left_next' not in structure:
            structure['left_next'] = {}
        if 'right_next' not in structure:
            structure['right_next'] = {}
        
        recurse_split(X_left, Y_left, structure=structure['left_next'], depth=depth+1)
        recurse_split(X_right, Y_right, structure=structure['right_next'], depth=depth+1)
        

    def get_best_X_split(self, X_region, Y_region):
        best_feat_i = 0
        best_feat_val = 0.0

        best_err = 9999999.9
        
        for feat_i in range(1, X_region.shape[1]-1):
            feat_min = X_region[:, feat_i].min()
            feat_max = X_region[:, feat_i].max()
            feat_steps = np.linspace(feat_min, feat_max, self.STEPS)

            for feat_step in feat_steps:
                X_Y = np.c_[X_region, Y_region]
                
                XY_left = X_Y[X_region[:, feat_i] < feat_step]
                XY_right = X_Y[X_region[:, feat_i] >= feat_step]

                region_err = self.get_region_MSE(XY_left[:, -1]) + self.get_region_MSE(XY_right[:, -1])

                if region_err < best_err:
                    best_err = region_err
                    best_feat_i = feat_i
                    best_feat_val = feat_step
        
        # Split X and Y based on best splitpoint
        X_Y = np.c_[X_region, Y_region]
        
        best_XY_left = X_Y[X_Y[:, best_feat_i] < best_feat_val]
        best_XY_right = X_Y[X_Y[:, best_feat_i] >= best_feat_val]

        X_left, Y_left = best_XY_left[:, :-1], best_XY_left[:, -1]
        X_right, Y_right = best_XY_right[:, :-1], best_XY_right[:, -1]
        
        return {'X_left': X_left, 'Y_left': Y_left, 
                'X_right': X_right, 'Y_right': Y_right, 
                'feat_i': best_feat_i, 'feat_val': best_feat_val, 'best_err': best_err}

    def get_region_MSE(self, Y):
        if len(Y) == 0: return 0
        Y_mean = Y.mean()
        return np.average(np.square(Y - Y_mean))


    def predict(self, X):
        pass