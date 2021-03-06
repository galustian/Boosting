import numpy as np
import numba as nb

class DecisionTreeRegressor:

    def __init__(self, tree_depth=3, min_datapoints=2, min_leaf_samples=1):
        #self.STEPS = 200
        self.DEPTH = tree_depth
        self.MIN_DATAPOINTS = min_datapoints
        self.MIN_LEAF_SAMPLES = min_leaf_samples
        self.structure = {}

    def fit(self, X, Y):
        # Give datapoints an ID
        X = np.c_[np.arange(len(X)), X]
        self.recurse_split(X, Y, structure=self.structure, depth=1)
    
    def recurse_split(self, X_region, Y_region, structure=None, depth=0):
        if 'END_NODE' in structure: 
            return

        split = self.get_best_X_split(X_region, Y_region)

        X_left, Y_left = split['X_left'], split['Y_left']
        X_right, Y_right = split['X_right'], split['Y_right']
        # TODO maybe variable importance?
        # best_err = split['best_err']
        structure['feat_i'] = split['feat_i']
        structure['feat_val'] = split['feat_val']
        
        # -1 if best_split not found (because of self.MIN_LEAF_SAMPLES)
        if split['best_err'] == -1:
            structure['left_next'] = {}
            structure['left_next']['END_NODE'] = True
            structure['left_next']['prediction'] = Y_region.mean()
            structure['right_next'] = {}
            structure['right_next']['END_NODE'] = True
            structure['right_next']['prediction'] = Y_region.mean()
            return

        #print("Depth:", depth)
        # If maximum Depth reached: construct right- and left endnodes
        if depth == self.DEPTH:
            structure['left_next'] = {}
            structure['left_next']['END_NODE'] = True
            structure['left_next']['prediction'] = Y_left.mean()
            structure['right_next'] = {}
            structure['right_next']['END_NODE'] = True
            structure['right_next']['prediction'] = Y_right.mean()
            return

        # If left of right less than MIN_DATAPOINTS, construct respective endnode
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
        
        self.recurse_split(X_left, Y_left, structure=structure['left_next'], depth=depth+1)
        self.recurse_split(X_right, Y_right, structure=structure['right_next'], depth=depth+1)
        
    #@nb.njit
    def get_best_X_split(self, X_region, Y_region):
        best_feat_i = 1
        best_feat_val = 0.0
        best_err = -1

        for feat_i in range(1, X_region.shape[1]):
            #feat_min = X_region[:, feat_i].min()
            #feat_max = X_region[:, feat_i].max()
            #feat_steps = np.linspace(feat_min, feat_max, self.STEPS)
            X_Y = np.c_[X_region, Y_region]
            #for feat_step in feat_steps:
            for x_i in range(len(X_region)):
                feat_step = X_region[x_i, feat_i]
                
                XY_left = X_Y[X_region[:, feat_i] < feat_step]
                XY_right = X_Y[X_region[:, feat_i] >= feat_step]

                if len(XY_left) < self.MIN_LEAF_SAMPLES or len(XY_right) < self.MIN_LEAF_SAMPLES:
                    continue

                #region_err = np.var(XY_left[:, -1]) + np.var(XY_right[:, -1])
                region_err = self.sum_of_squared_err(XY_left[:, -1]) + self.sum_of_squared_err(XY_right[:, -1])

                if region_err < best_err or best_err == -1:
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

    def sum_of_squared_err(self, Y):
        return np.sum(np.square(Y - Y.mean()))

    def predict(self, X):
        Y = []
        for i in range(len(X)):
            y_hat = self.predict_sample(X[i])
            Y.append(y_hat)
        
        return np.array(Y)

    def predict_sample(self, X, struct=None):
        if struct == None:
            X = np.r_[0, X]
            struct = self.structure
        
        if 'END_NODE' in struct:
            return struct['prediction']
            
        if X[struct['feat_i']] < struct['feat_val']:
            return self.predict_sample(X, struct=struct['left_next'])
        else:
            return self.predict_sample(X, struct=struct['right_next'])
