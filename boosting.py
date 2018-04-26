import numba as nb
import numpy as np

spec1 = [
    ('n_classifiers', nb.uint16),
    ('extra_trees', nb.boolean),
]

# @nb.jitclass(spec1)
class AdaBoostClassifier:
    
    def __init__(self, n_classifiers=50, extra_trees=False):
        self.n_classifiers = n_classifiers
        self.extra_trees = extra_trees
        self.classifiers = []
        self.classifier_w = []
    
    # X must be a numpy array where each row is a datapoint
    def fit(self, X, Y):
        X_IDs = np.arange(0, X.shape[0])
        X_Weights = np.array([1 / X.shape[0] for i in range(X.shape[0])])

        X = np.c_[X_IDs, X]

        best_classifiers = []

        for i in range(n_classifiers):
            stump_classifier = DecisionTreeStump(extra_trees)
            stump_classifier.fit(X, Y, X_Weights) # compute best decision-tree stump

            self.classifiers.append(stump_classifier)
            self.classifier_w.append(self.compute_alpha(stump_classifier.total_error))
            
            # TODO compute new X_Weights


    @staticmethod
    def compute_alpha(err):
        return np.log((1 - err) / err) / 2

    def predict(self, X):
        pass

spec2 = [
    ('extra_trees', nb.boolean),
    ('feat_i', nb.uint32),
    ('feat_size', nb.float32),
    ('abs_error', nb.float32),
    ('total_error', nb.float32),
    ('wrong_idx', nb.int64[:])
]

@nb.jitclass(spec2)
class DecisionTreeStump:

    def __init__(self, extra_trees=False):
        self.extra_trees = extra_trees

    # Compute the best classifier
    def fit(self, X, Y, X_Weights):
        n_feat = X.shape[1]-1

        best_feat_i = 0
        best_feat_size = 0.0  # smaller than feat_size (left side): -ones, on the right: ones
        furthest_from_half_error = 0.0      # absolute distance from 1/2 (close to 0 => bad, close to 1/2 => good!)
        total_error = 0.0 # needed for computing alpha

        wrong_idx = [0] # which x_i's are classified wrong
        wrong_idx.pop()
        
        # TODO use nb.prange
        # TODO (maybe): make algorithm more efficient
        for feat_i in range(1, n_feat):
            # Compute Error for each possible tree => choose best stump
            for n in range(X.shape[0]):
                n_feat_val = X[n, feat_i]
                
                stump_left = X[X[:, feat_i] < n_feat_val]
                stump_right = X[X[:, feat_i] >= n_feat_val]
                
                # sum up all weights for misclassified samples
                error = 0.0
                temp_wrong_idx = [0]
                temp_wrong_idx.pop()

                for x_i in stump_left[:, 0]:
                    x_ii = int(x_i)
                    if Y[x_ii] != -1:
                        error += X_Weights[x_ii]
                        temp_wrong_idx.append(x_ii)
                
                for x_i in stump_right[:, 0]:
                    x_ii = int(x_i)
                    if Y[x_ii] != 1:
                        error += X_Weights[x_ii]
                        temp_wrong_idx.append(x_ii)

                if abs(error - 1/2) > furthest_from_half_error:
                    furthest_from_half_error = abs(error - 1/2)
                    best_feat_i = feat_i
                    best_feat_size = n_feat_val
                    total_error = error
                    wrong_idx = temp_wrong_idx
        
        self.feat_i = best_feat_i
        self.feat_size = best_feat_size
        self.total_error = total_error
        self.wrong_idx = np.array(wrong_idx)

    def predict(self, X):
        pass
    