from utils.check_subscriptions import prepare_text_df, OneHotEncoder
from config import BUCKET_NAME
import pandas as pd
import boto3
from config import RISK_SCORE_COL, s3_MODELS_FOLDER


def forecast_batch(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data)  # Dataframe with feedback text
    df = prepare_text_df(df)
    ohe_model = load_from_S3('ohencoder.bin')
    df = pd.DataFrame(ohe_model.transform(df['processed_message']),
                      columns=ohe_model.classes_,
                      index=df['processed_message'].index)
    df_ohe = ohe_model.convert(text_data=df, use_local_model=True)

    model_trained = load_from_S3('reg_trained.bin')
    y_fcst = model_trained.predict(df_ohe)
    df[RISK_SCORE_COL] = y_fcst
    return df

def load_from_S3(filename: str):
    return boto3.resource('s3').Object(bucket_name=BUCKET_NAME, key=s3_MODELS_FOLDER + '/' + filename).load()