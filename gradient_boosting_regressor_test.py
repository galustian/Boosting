import numpy as np
import pandas as pd
from gradient_boosting_regressor import GradientBoostingRegressor

def gradient_boosting_regressor():
    reg = GradientBoostingRegressor(iterations=20, tree_depth=3)
    
    df = pd.read_csv('USA_Housing.csv').sample(frac=1, random_state=1)
    df = df.iloc[:, :-1]

    X_train, Y_train = df.iloc[:int(len(df) / 2), :-1].as_matrix(), df.iloc[:int(len(df) / 2), -1].as_matrix()
    X_test, Y_test = df.iloc[int(len(df) / 2):, :-1].as_matrix(), df.iloc[int(len(df) / 2):, -1].as_matrix()

    reg.fit(X_train, Y_train)
    Y_hat = reg.predict(X_test)

    # Compute MSE
    print("MSE:", np.average(np.square(Y_test - Y_hat)))
