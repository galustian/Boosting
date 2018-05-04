import numba as nb
import numpy as np


class AdaBoostClassifier:
    
    def __init__(self, n_classifiers=50):
        self.n_classifiers = n_classifiers
        self.classifier_alpha = []
        self.classifiers = []
    
    # X must be a numpy array where each row is a datapoint
    def fit(self, X, Y):
        X_IDs = np.arange(0, X.shape[0])
        X_Weights = np.array([1 / X.shape[0] for i in range(X.shape[0])])

        X = np.c_[X_IDs, X]

        for i in range(self.n_classifiers):
            stump_classifier = DecisionTreeStump()
            stump_classifier.fit(X, Y, X_Weights) # compute best decision-tree stump

            # if total_error of stump_classifier is close to 0 or 1, it is stuck in an infinite loop
            # (alphas are 0 or infinite and weights don't update anymore)
            self.classifiers.append(stump_classifier)

            alpha = compute_alpha(stump_classifier.total_error)
            self.classifier_alpha.append(alpha)
            
            if abs(abs(stump_classifier.total_error - 1/2) - 1/2) < 1e-6:
                self.n_classifiers = i+1
                break
            
            # If error-rate of whole boosting classifier is 0: break
            if np.array_equal(Y, self.predict(X)):
                self.n_classifiers = i+1
                break
            
            X_Weights = update_weights(X_Weights, stump_classifier)

    def predict(self, X):
        prediction_2D_array = []

        for i, clf in enumerate(self.classifiers):
            predictions = self.classifier_alpha[i] * clf.predict(X)
            prediction_2D_array.append(predictions)
        
        prediction_2D_array = np.array(prediction_2D_array).T
        
        assert prediction_2D_array.shape[0] == X.shape[0]
        assert prediction_2D_array.shape[1] == len(self.classifiers)
        
        return np.sign(prediction_2D_array.sum(axis=1)).astype(np.int8)

# ----------------------------------------- HELPERS -----------------------------------------

#@nb.njit
def update_weights(X_Weights, classifier):
    for x_i in range(len(X_Weights)):
        if x_i in classifier.wrong_idx:
            X_Weights[x_i] = X_Weights[x_i] * (1/2) * (1 / classifier.total_error)
        else:
            X_Weights[x_i] = X_Weights[x_i] * (1/2) * (1 / (1 - classifier.total_error))
    
    return X_Weights

@nb.njit
def compute_alpha(err):
    if err == 1.0:
        return -7
    if err == 0.0:
        return 7
    return np.log((1 - err) / err) / 2


spec2 = [
    ('feat_i', nb.uint32),
    ('feat_size', nb.float32),
    ('STEPS', nb.uint16),
    ('total_error', nb.float32),
    ('wrong_idx', nb.int64[:])
]

@nb.jitclass(spec2)
class DecisionTreeStump:
    def __init__(self):
        self.STEPS = 25

    # Compute the best classifier
    def fit(self, X, Y, X_Weights):
        n_feat = X.shape[1]-1

        best_feat_i = 0
        best_feat_size = 0.0  # smaller than feat_size (left side): -ones, on the right: ones
        furthest_from_half_error = 0.0  # absolute distance from 1/2 (close to 0 => bad, close to 1/2 => good!)
        total_error = 0.0 # needed for computing alpha

        wrong_idx = [0] # which x_i's are classified wrong
        wrong_idx.pop()
        
        # TODO use nb.prange
        # TODO (maybe): make algorithm more efficient
        for feat_i in range(1, n_feat):
            # Compute Error for each possible tree => choose best stump
            feat_min = X[:, feat_i].min()
            feat_max = X[:, feat_i].max()

            feat_steps = np.linspace(feat_min, feat_max, self.STEPS)

            for step_i in range(len(feat_steps)):
                feat_size = feat_steps[step_i]
                
                stump_left = X[X[:, feat_i] < feat_size]
                stump_right = X[X[:, feat_i] >= feat_size]
                
                # Sum up all weights for misclassified samples
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
                    total_error = error
                    wrong_idx = temp_wrong_idx
                    
                    best_feat_i = feat_i
                    best_feat_size = feat_size
        
        self.feat_i = best_feat_i
        self.feat_size = best_feat_size
        self.total_error = total_error
        self.wrong_idx = np.array(wrong_idx)

    def predict_sample(self, x):
        if len(x.shape) != 1:
            raise TypeError('predict_sample takes one-dimensional numpy arrays only. Dim != 1')
        
        if x[self.feat_i] < self.feat_size:
            return -1
        return 1

    def predict(self, X):
        if len(X) == 1:
            raise TypeError('predict takes two-dimensional numpy arrays only. Dim: 1')

        Y_hat = [0]
        Y_hat.pop()

        for i in range(len(X)):
            if X[i][self.feat_i] < self.feat_size:
                Y_hat.append(-1)
            else:
                Y_hat.append(1)
        
        return np.array(Y_hat, dtype=np.int8)
    
