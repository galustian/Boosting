import numpy as np
from adaboost import AdaBoostClassifier, DecisionTreeStump


def test_adaboost_classifier():
    X = np.array([
        [7.2, 8, 45],
        [3.44, 7, 48],
        [2.19, 6, 1],
        [2.18, 6.1, 129],
        [2.18, 5, 19],
    ])
    Y = np.array([-1, -1, 1, 1, 1], dtype=np.int)
    
    clf = AdaBoostClassifier()
    clf.fit(X, Y)
    # must classify all correct with 1 classifier
    assert clf.n_classifiers == 1

    X = np.array([
        [1.2, 8, 45],
        [3.44, 7, 48],
        [2.19, 6, 1],
        [2.20, 5, 129],
        [1.35, 15, 19],
    ])
    Y = np.array([-1, -1, -1, 1, 1], dtype=np.int)

    clf = AdaBoostClassifier(50)
    clf.fit(X, Y)

    assert clf.n_classifiers != 1
    assert clf.n_classifiers < 6


def test_decisiontree_stump_fit():
    stump = DecisionTreeStump()

    X = np.array([
        [0, 7.2, 8, 45],
        [1, 3.44, 7, 48],
        [2, 2.19, 6, 1],
        [3, 2.18, 6.1, 129],
        [4, 2.18, 5, 19],
    ])
    Y = np.array([-1, -1, 1, 1, 1], dtype=np.int)
    X_Weights = np.array([1 / X.shape[0] for i in range(X.shape[0])])

    stump.fit(X, Y, X_Weights)

    assert stump.feat_i == 1
    assert np.around(stump.feat_size, 2) == 3.44
    assert stump.total_error == 1.0

    X = np.array([
        [0, 1.2, 8, 45],
        [1, 3.44, 7, 48],
        [2, 2.19, 6, 1],
        [3, 2.20, 5, 129],
        [4, 1.35, 15, 19],
    ])
    Y = np.array([-1, -1, -1, 1, 1], dtype=np.int)

    stump = DecisionTreeStump()
    stump.fit(X, Y, X_Weights)

    assert stump.feat_i == 2
    assert np.around(stump.feat_size, 0) == 6
    assert np.around(stump.total_error, 1) == 0.8


def test_decisiontree_stump_predict():
    X = np.array([
        [0, 1.2, 8, 45],
        [1, 3.44, 7, 48],
        [2, 2.19, 6, 1],
        [3, 2.20, 5, 129],
        [4, 1.35, 15, 19],
    ])
    Y = np.array([-1, -1, -1, 1, 1], dtype=np.int)
    X_Weights = np.array([1 / X.shape[0] for i in range(X.shape[0])])

    stump = DecisionTreeStump()
    stump.fit(X, Y, X_Weights)
    
    x_hat = np.array([24554, 2.1, 6, 1])
    y_hat = stump.predict_sample(x_hat)
    assert y_hat == 1

    Y_hat = stump.predict(X)
    
    assert Y_hat[0] == 1
    assert Y_hat[1] == 1
    assert Y_hat[2] == 1
    assert Y_hat[3] == -1
    assert Y_hat[4] == 1