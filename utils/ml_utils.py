import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
import xgboost
import numpy as np

import logging
FORMAT = '%(asctime) %(user) %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger('ml_utils')

def get_optimum_Kmean(train_data, max_clusters: int=10, show_graph: bool=False):
    distorsions = []
    n_cluster = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(train_data)
        n_cluster.append(k)
        distorsions.append(kmeans.inertia_)


    slope = np.array([abs(distorsions[n] - distorsions[n-1]) for n in range(1, len(distorsions))])  # get lines slope
    min_slope = np.argmin(slope)  # position of the line that its slope decrease the least
    optimum_clusters = min_slope + 2
    optimum_kmeans = KMeans(n_clusters=optimum_clusters)
    optimum_kmeans.fit(train_data)

    if show_graph:
        fig = plt.figure(figsize=(15, 5))
        plt.plot(n_cluster, distorsions)
        plt.grid(True)
        plt.title('Elbow curve')
        plt.show()

    return optimum_clusters, optimum_kmeans


def get_optimum_classification(train_data, classes_data: pd.DataFrame, train_pct: float=0.8):
    X_train, X_test, y_train, y_test = train_test_split(train_data,
                                                        classes_data,
                                                        test_size = (1-train_pct),
                                                        random_state = 1988,
                                                        stratify=classes_data)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    logger.info(confusion_matrix(y_test, predictions, labels=[1,2,3,4,5]))
    return  model

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

def get_optimum_xgboost(train_data, values_data: pd.DataFrame, train_pct: float=0.8):
    X_train, X_test, y_train, y_test = train_test_split(train_data,
                                                        values_data,
                                                        test_size = (1-train_pct),
                                                        random_state = 1988)
    model = xgboost.XGBRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    logger.info(f'mean abs error: {mean_absolute_error(y_test, predictions)}')
    logger.info(f'root mean sqrt error: {mean_squared_error(y_test, predictions, squared=False)}')
    return model

def split_features_y(df: pd.DataFrame, col_to_predict: str = 'Y'):
    cols_feat = [col for col in df.columns if col != col_to_predict]
    return df[cols_feat], df[[col_to_predict]]


def ohe(df: pd.DataFrame):
    ohe = df.select_dtypes(include='object')
    cols_2_ohe = ohe.columns
    ohe = pd.get_dummies(ohe)
    df_no_ohe = df[[col for col in df.columns if col not in cols_2_ohe]]
    output = pd.concat([df_no_ohe, ohe], axis=1)
    return output