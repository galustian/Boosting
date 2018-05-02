import numpy as np
import pandas as pd
from gradient_boosting_regressor import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor as GradientBoostingRegressor2

def test_gradient_boosting_regressor():
    reg = GradientBoostingRegressor(iterations=50, learning_rate=0.065, tree_depth=5)
    
    df = pd.read_csv('boston.csv').sample(frac=1, random_state=14).iloc[:, 1:]

    train_len = int(len(df) / 2)

    X_train, Y_train = df.iloc[:train_len, :-1].as_matrix(), df.iloc[:train_len, -1].as_matrix()
    X_test, Y_test = df.iloc[train_len:, :-1].as_matrix(), df.iloc[train_len:, -1].as_matrix()
    
    reg.fit(X_train, Y_train)
    Y_hat = reg.predict(X_test)

    dec_reg = DecisionTreeRegressor(max_depth=15, min_samples_split=10)
    dec_reg.fit(X_train, Y_train)
    Y_hat2 = dec_reg.predict(X_test)

    grad2_reg = GradientBoostingRegressor2(n_estimators=50, learning_rate=0.065, max_depth=5)
    grad2_reg.fit(X_train, Y_train)
    Y_hat3 = grad2_reg.predict(X_test)
    
    # Compute MSE
    print("Grad-MSE:", np.var(Y_test - Y_hat))
    print("Dec-MSE:", np.var(Y_test - Y_hat2))
    print("Grad2-MSE:", np.var(Y_test - Y_hat3))

    assert np.var(Y_test - Y_hat) < np.var(Y_test - Y_hat2)
