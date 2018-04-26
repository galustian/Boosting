import numpy as np
from boosting import DecisionTreeStump

def test_decisiontree_stump():
    stump = DecisionTreeStump(extra_trees=False)

    X = np.array([
        [0, 7.2, 8, 45],
        [1, 3.44, 7, 48],
        [2, 2.19, 6, 1],
        [3, 2.18, 6.1, 129],
        [4, 2.18, 5, 19],
    ])
    Y = np.array([0, 0, 1, 1, 1], dtype=np.int)
    
    X_Weights = np.array([1 / X.shape[0] for i in range(X.shape[0])])

    stump.fit(X, Y, X_Weights)

    assert stump.feat_i == 1
    assert np.around(stump.feat_size, 2) == 3.44
    assert stump.abs_error == 0.5

    X = np.array([
        [0, 1.2, 8, 45],
        [1, 3.44, 7, 48],
        [2, 2.19, 6, 1],
        [3, 2.20, 5, 129],
        [4, 1.35, 15, 19],
    ])
    Y = np.array([0, 0, 0, 1, 1], dtype=np.int)

    stump = DecisionTreeStump(extra_trees=False)
    stump.fit(X, Y, X_Weights)

    assert stump.feat_i == 2
    assert np.around(stump.feat_size, 0) == 6
    assert np.around(stump.abs_error, 1) == abs(1 / X.shape[0] - 1/2)