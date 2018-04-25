import numba as nb
import numpy as np

spec1 = [
    ('n_classifiers', nb.uint16),
    ('extra_trees', nb.boolean)
]

@nb.jitclass(spec1)
class AdaBoostClassifier:
    
    def __init__(self, n_classifiers=50, extra_trees=False):
        self.n_classifiers = n_classifiers
        self.extra_trees = extra_trees
    
    # X must be a numpy array where each row is a datapoint
    def fit(self, X, Y):
        X_IDs = np.array([i for i in range(X.shape[0])])
        X_Weights = np.array([1 / X.shape[0] for i in range(X.shape[0])])

        X = np.c_[X_IDs, X]

        best_classifiers = []

        for i in range(n_classifiers):
            stump_classifier = DecisionTreeStump(extra_trees)


    def predict(self, X):
        pass

spec2 = [
    ('extra_trees', nb.boolean)
]

@nb.jitclass()
class DecisionTreeStump:

    def __init__(self, extra_trees=False):
        self.extra_trees = extra_trees

    # Compute the best classifier
    def fit(self, X, X_Weights, Y)
        n_feat = X.shape[1]

        best_feat_i = 0
        best_feat_size = 0.0  # smaller than feat_size (on the left side): zeros, on the right: ones
        best_error = 0.0
        
        # TODO (maybe): make algorithm more efficient
        for feat_i in range(n_feat):
            # Compute Error for each possible tree => choose best stump
            for n in range(X.shape[0]):
                n_feat_val = X[n, feat_i]
                
                # err_right = 1 - err_left
                stump_left = X[X[:, feat_i] < n_feat_val]
                
                # sum up all weights for misclassified samples
                error = 0.0
                for x_i in range(stump_left.shape[0]):
                    if Y[x_i] != 0:
                        error += X_Weights[x_i]

                if error < best_error:
                    best_error = error
                    best_feat_i = feat_i
                    best_feat_size = n_feat_val
        
        


    def predict(self, X)
        pass
    