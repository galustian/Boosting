import numpy as np
import pandas as pd
from decision_tree_regressor import DecisionTreeRegressor

'''def test_get_best_X_split():
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
    '''
def test_predict():
    reg = DecisionTreeRegressor(tree_depth=5)
    
    df = pd.read_csv('boston.csv').sample(frac=1, random_state=34)
    df = df.iloc[:, 1:]
    #df['Price'] = df.iloc[:, -1] - df.iloc[:, -1].mean()
    #df['Price'] = df.iloc[:, -1] / df.iloc[:, -1].std()

    X_train, Y_train = df.iloc[:int(len(df) / 2.6), :-1].as_matrix(), df.iloc[:int(len(df) / 2.6), -1].as_matrix()
    X_test, Y_test = df.iloc[int(len(df) / 2.6):, :-1].as_matrix(), df.iloc[int(len(df) / 2.6):, -1].as_matrix()

    reg.fit(X_train, Y_train)
    Y_hat = reg.predict(X_test)

    # Compute MSE
    print("MSE:", np.average(np.square(Y_test - Y_hat)))
    
    for i in range(len(Y_test)):
        print(Y_test[i], "pred:", Y_hat[i])
    print(reg.structure)

    ''' reg = DecisionTreeRegressor(tree_depth=1, min_datapoints=3)
    X = np.array([
        [4.5, 3, 4],
        [4.55, 0.3, 5],
        [6.8, 0.102, 6],
        [6.99, 99, 7],
        [9.01, 102, 8],
        [19.01, 4, 7.5],
    ])
    Y = np.array([2.4, 2.45, 3.57, 4.5, 4.9, 5.5])

    reg.fit(X, Y)

    assert reg.predict_sample(np.array([4.52, 45, 455.03])) == Y[:3].mean()

    reg = DecisionTreeRegressor(tree_depth=2, min_datapoints=3)
    X = np.array([
        [4.5, 3, 4],
        [4.55, 0.3, 5],
        [6.8, 0.102, 6],
        [6.99, 99, 7],
        [9.01, 112, 8],
        [19.01, 4, 7.5],
        [20.01, 120, 0.5],
        [21.01, 133, 9.5],
        [18.21, 145, 7.5],
        [24.01, 150, 1.5],
        [23.31, 155, 3.5],
    ])
    Y = np.array([2.4, 2.45, 3.57, 44.5, 44.9, 45.5, 50.3, 52, 55, 56, 57])

    reg.fit(X, Y)

    assert reg.predict_sample(np.array([18, 165, 4])) == Y[6:].mean()
    '''