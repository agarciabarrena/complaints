import os
import pickle

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from utils.text_handling import text_preparation
import logging
logger = logging.getLogger('check_subscription')

MODELS_FOLDER = 'models'


def prepare_text_df(df, lang, model_ohe_fit: bool = False):
    lang_dict = {'EN': 'english', 'FR': 'french', 'TK': 'turkish', 'ES': 'spanish'}
    df = df_cleaning(df)
    # TODO Add translation here
    df = text_preparation(df, lang_dict.get(lang))
    if model_ohe_fit:
        mlb = MultiLabelBinarizer()
        mlb = mlb.fit(df['processed_message'])
        save_model(mlb, 'ohencoder')
    else:
        mlb = load_model('ohencoder')
    df_text = pd.DataFrame(mlb.transform(df['processed_message']), columns=mlb.classes_, index=df['processed_message'].index)
    return df_text


def df_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = df.shape[0]
    if 'customer_care_id' in df.columns:
        df.drop('customer_care_id', axis=1, inplace=True)
    df = df[df['message'] != ""]  # Clean empty
    df = df.assign(timestamp=pd.to_datetime(df['timestamp']),
                   message=df['message'].str.lower())
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