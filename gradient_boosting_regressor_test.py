import numpy as np
from gradient_boosting_regressor import DecisionTreeRegressor

def test_get_best_X_split():
    reg = DecisionTreeRegressor(tree_depth=3, min_datapoints=3)

    X_region = np.array([
        [0, 4.5, 3, 4],
        [1, 4.55, 0.3, 5],
        [2, 6.8, 0.102, 6],
        [3, 6.99, 99, 7],
        [4, 9.01, 102, 8],
        [5, 19.01, 4, 7.5],
    ])
    Y_region = np.array([2.4, 2.45, 3.57, 4.5, 4.9, 5.5])
    
    split = reg.get_best_X_split(X_region, Y_region)
    
    # ASSERT
    X_left = np.array([
        [0, 4.5, 3, 4],
        [1, 4.55, 0.3, 5],
        [2, 6.8, 0.102, 6],
    ])
    X_right = np.array([
        [3, 6.99, 99, 7],
        [4, 9.01, 102, 8],
        [5, 19.01, 4, 7.5],
    ])
    
    assert np.array_equiv(split['X_left'], X_left)
    assert np.array_equiv(split['X_right'], X_right)
    assert 6.8 <= split['feat_val'] <= 6.99 