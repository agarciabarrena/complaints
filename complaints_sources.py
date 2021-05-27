from connectors.db_connector import RedshiftConnector
import pandas as pd


with open('sql/complaint_sources.sql', 'r') as file:
    query = file.read()

conn = RedshiftConnector()
df = conn.query_df(query)

df_tag = pd.read_csv('data_raw/complaints_fcst_and_scored.csv', sep=',')  #'data_raw/complaints_tagged_reg.csv', sep=';')

d = df_tag.merge(df,'left', 'customer_care_id')

pd.set_option('display.max_rows', None)
# print(d.groupby('google_placement_name').agg({'risk_score_combined': 'mean', 'customer_care_id': 'count'}))
for field in ['google_tid', 'vertical', 'banner']:
    dt = d.groupby(['country_code', field]).agg({'risk_score': 'mean', 'customer_care_id': 'count'})
    # print(dt)
    print(dt[(dt['customer_care_id'] >= 2) & (dt['risk_score'] >= 7)])