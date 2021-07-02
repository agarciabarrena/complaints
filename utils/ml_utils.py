import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import logger

def get_optimum_regression(train_data, values_data: pd.DataFrame, train_pct: float=0.8):
    X_train, X_test, y_train, y_test = train_test_split(train_data,
                                                        values_data,
                                                        test_size = (1-train_pct),
                                                        random_state = 1988)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    logger.info(f'mean abs error: {mean_absolute_error(y_test, predictions)}')
    logger.info(f'root mean sqrt error: {mean_squared_error(y_test, predictions, squared=False)}')
    return model


def ohe(df: pd.DataFrame):
    ohe = df.select_dtypes(include='object')
    cols_2_ohe = ohe.columns
    ohe = pd.get_dummies(ohe)
    df_no_ohe = df[[col for col in df.columns if col not in cols_2_ohe]]
    output = pd.concat([df_no_ohe, ohe], axis=1)
    return output