import os
import pickle

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from config import MODELS_FOLDER, logger
from datetime import datetime

from utils.connectors.db_connector import RedshiftConnector
from utils.text_handling import text_preparation
from config import LANGUAGE_COL, training_data_path



def prepare_text_df(df):
    df = df_cleaning(df)
    df = text_preparation(df)
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
        if not self.ohe_model and not use_local_model:
            logger.error("Please train the model first or select the use_local_model option instead")
        if use_local_model:
            ohe_model = self.__load_local()
        else:
            ohe_model = self.ohe_model
        df_text = pd.DataFrame(ohe_model.transform(text_data['processed_message']),
                           columns=ohe_model.classes_,
                           index=text_data['processed_message'].index)
        return df_text


class Complains():
    def forecast_new_feedbacks(self):
        '''
        Send the feedback/s to the API to get the scores and the translations if needed
        '''
        df = self.__extract_new_feedbacks()
        # TODO make connection to the API
        return None

    def __extract_last_report_date(self):
        '''
        Query the database to obtain the last execution date so we can find new feedbacks since last processing
        '''
        query = ("""SELECT MAX(insert_timestamp) as execution_date 
                from customer_care_flow_events where action='feedback_forecasted'""")
        conn = RedshiftConnector()
        df = conn.query_df(query)
        max_date = df['execution_date'][0]
        return max_date

    def __extract_new_feedbacks(self):
        last_date = self.__extract_last_report_date()
        with open('sql/complaints_new.sql', 'r') as f:
            query = f.read()
        query = query.format(last_execution_timestamp=last_date)
        conn = RedshiftConnector()
        df = conn.query_df(query)
        if df.empty:
            logger.info(f'There is no new feedback after {last_date}')
            exit()
        return df

    def save_in_redshift_table(self):
        return None


def df_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'customer_care_id' in df.columns:
        df.drop('customer_care_id', axis=1, inplace=True)
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
        df = pd.read_csv(training_data_path, sep=';')
        df[LANGUAGE_COL] = df[LANGUAGE_COL].str.lower()
    else:
        with open('sql/complaints_new.sql', 'r') as file:
            query = file.read()
        conn = RedshiftConnector()
        df = conn.query_df(query)
    return df

def manual_add_review(review: str, score: float, language: str='en'):
    df = load_data(training_data=True)
    max_num_manual_review = df['customer_care_id'].str.split('manual_upload_').str[1].astype(float).max()
    row_2_append = pd.DataFrame({'customer_care_id': [f'manual_upload_{int(max_num_manual_review+1)}'],
                                 'timestamp': [datetime.now()],
                                 'message': [review],
                                 'language': [language],
                                 'risk_score':[0],
                                 'wes_score': [0],
                                 'risk_score_combined': [score]})
    df = df.append(row_2_append, ignore_index=True)
    df.to_csv(training_data_path, sep=';', index=False)
    return None