from utils.ml_utils import get_optimum_regression
from utils.check_subscriptions import prepare_text_df,\
    save_model, load_model, OneHotEncoder, load_data, manual_add_review, Complains
from dotenv import load_dotenv
from config import logger, RISK_SCORE_COL
import pandas as pd


load_dotenv()


def train_models():
    df = load_data(training_data=True)
    df = prepare_text_df(df)
    ohe = OneHotEncoder()
    ohe.train(text_data=df, save_local=True)
    df_ohe = ohe.convert(text_data=df)

    df_y = df.loc[df_ohe.index, 'risk_score_combined']
    model_trained = get_optimum_regression(df_ohe, df_y)
    save_model(model_trained, 'reg_trained')
    logger.debug('OHE and Regression model train and saved locally')


def forecast_batch():
    df = load_data(training_data=False)
    df = prepare_text_df(df)
    ohe = OneHotEncoder()
    df_ohe = ohe.convert(text_data=df, use_local_model=True)

    model_trained = load_model('reg_trained')
    y_fcst = model_trained.predict(df_ohe)
    df[RISK_SCORE_COL] = y_fcst
    return df


def forecast_single(text: str):
    df = pd.DataFrame({'message': [text], 'language': ['single_input']})
    df = prepare_text_df(df)
    ohe = OneHotEncoder()
    df_ohe = ohe.convert(text_data=df, use_local_model=True)

    model_trained = load_model('reg_trained')
    y_fcst = model_trained.predict(df_ohe)
    return y_fcst

def add_train_review(review: str,score: float, language: str='en'):
    manual_add_review(review=review, score=score, language=language)

c = Complains()
c.forecast_new_feedbacks()
