from utils.connectors.db_connector import RedshiftConnector
import pandas as pd
from utils.ml_utils import get_optimum_regression
from utils.check_subscriptions import prepare_text_df, save_model, load_model
from config import logger


lang = 'EN'

df_scored = pd.read_csv('data_raw/complaints_tagged_reg.csv', sep=';')

df_train = prepare_text_df(df_scored, 'EN', model_ohe_fit=True)

df_y = df_scored.loc[df_train.index,'risk_score_combined']
model_trained = get_optimum_regression(df_train, df_y)
save_model(model_trained, 'reg_trained')


logger.info('SECOND part')
with open('sql/complaints_analysis.sql') as f:
    query = f.read()
connector = RedshiftConnector()
df = connector.query_df(query=query)
logger.info('data loaded')
text = prepare_text_df(df, 'EN', model_ohe_fit=False)
model_trained = load_model('reg_trained')
y = model_trained.predict(text)

new_records = df.loc[text.index,['customer_care_id','timestamp', 'message']]
new_records['fcs_risk_score'] = y
new_records = new_records[new_records.timestamp > df_scored.timestamp.max()]

total_df = pd.concat([df_scored[['customer_care_id', 'timestamp', 'message', 'risk_score_combined']], new_records]
                     , axis=0)
total_df = total_df.assign(risk_score=total_df.apply(lambda x: x['risk_score_combined'] if pd.isna(x['fcs_risk_score']) else x['fcs_risk_score'],axis=1))
total_df.drop(columns=['risk_score_combined', 'fcs_risk_score'], inplace=True)
# total_df.to_csv('data_raw/complaints_fcst_and_scored.csv', index=False)

