import os
import pickle

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from config import MODELS_FOLDER, logger

from utils.connectors.db_connector import RedshiftConnector
from utils.text_handling import text_preparation



def prepare_text_df(df, text_column: str = 'message'):
    df = df_cleaning(df, text_column)
    df = text_preparation(df, 'english')
    return df


class OneHotEncoder():
    def __init__(self):
        self.ohe_model = False

    def __load_local(self):
        mlb = load_model('ohencoder')
        return mlb

    def __save_local(self, model):
        save_model(model, 'ohencoder')

    def train(self, text_data: pd.DataFrame=pd.DataFrame(), save_local: bool=False):
        assert not text_data.empty
        assert 'processed_message' in text_data.columns
        mlb = MultiLabelBinarizer()
        mlb = mlb.fit(text_data['processed_message'])
        if save_local:
            self.__save_local(mlb)
        self.ohe_model = mlb
        return mlb

    def convert(self, text_data: pd.DataFrame, use_local_model: bool=False):
        if not self.ohe_model:
            logger.error("Please train the model first or select the use_local_model option instead")
        if use_local_model:
            ohe_model = self.__load_local()
        else:
            ohe_model = self.ohe_model
        df_text = pd.DataFrame(ohe_model.transform(text_data['processed_message']),
                           columns=ohe_model.classes_,
                           index=text_data['processed_message'].index)
        return df_text

def df_cleaning(df: pd.DataFrame, text_column: str='message') -> pd.DataFrame:
    df = df.copy()
    n = df.shape[0]  # Save original number of reviews
    if 'customer_care_id' in df.columns:
        df.drop('customer_care_id', axis=1, inplace=True)
    df = df[df[text_column] != ""]  # Clean empty
    df = df.assign(message=df[text_column].str.lower())
    logger.info(f'Original number of reviews: {n}\nNumber of reviews after cleaning: {df.shape[0]}')
    return df


def save_model(object, name: str) -> None:
    with open(os.path.join(MODELS_FOLDER, f'{name}.bin'), 'wb') as file:
        pickle.dump(object, file)
    return None

def load_model(name: str):
    with open(os.path.join(MODELS_FOLDER, f'{name}.bin'), 'rb') as file:
        model = pickle.load(file)
    return model


def load_data(training_data: bool=False):
    if training_data:
        df = pd.read_csv('data_raw/complaints_tagged_reg.csv', sep=';')
    else:
        # TODO add query for future values with language modification made by Utkarsh
        with open('sql/complaints_analysis.sql', 'r') as file:
            query = file.read()
        conn = RedshiftConnector()
        df = conn.query_df(query)
    return df