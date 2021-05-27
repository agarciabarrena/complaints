from os import getenv
from datetime import datetime
import psycopg2
import logging
import boto3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import pandas as pd
logger = logging.getLogger('Connectors')

class RedshiftConnector:
    def __init__(self, verbose=False):
        self.credentials = getenv('CONNECTION_STRING')
        self.connector = psycopg2.connect(self.credentials)
        self.cursor = self.connector.cursor()
        self.verbose = verbose

    def __query_db(self, query):
        self.cursor.execute(query)
        colnames = [desc[0] for desc in self.cursor.description]
        return self.cursor.fetchall(), colnames

    def query_df(self, query):
        if self.verbose:
            print(query + '\n')
        raw_data, colnames = self.__query_db(query)
        df = pd.DataFrame(raw_data, columns=colnames)
        return df

    def get_connector(self):
        return self.connector

    def upload_df(self, df: pd.DataFrame, table_name: str, schema=None, mode='fail') -> None:
        engine = create_engine(self.credentials)
        Session = sessionmaker(bind=engine)
        session = Session()
        logger.info(f'Uploading {df.shape} dataframe to table {schema + "."}{table_name}')
        df.to_sql(name=table_name, con=session.get_bind(), schema=schema, index=False, if_exists=mode, method='multi')


    def copy_df_to_redshift(self, data: pd.DataFrame, table_name: str) -> None:
        filename = f'{datetime.now().strftime("%Y-%m-%d")}/{table_name}-{datetime.now().strftime("%H:%M:%S")}.csv'
        logger.info(f"Caching dataframe to S3 at: {getenv('S3_BUCKET_NAME')}/{filename}")
        boto3.resource('s3').Bucket(getenv('S3_BUCKET_NAME')).put_object(
            Body=data.to_csv(index=False),
            ContentType='text/csv',
            Key=filename,
        )
        logger.info("Uploading dataframe to redshift")
        self.cursor.execute(
            "COPY {table_name} (\"{columns}\") FROM '{s3_path}' IAM_ROLE '{iam_role}' CSV TRIMBLANKS TRUNCATECOLUMNS BLANKSASNULL EMPTYASNULL IGNOREHEADER 1".format(
                table_name=table_name,
                columns='\",\"'.join(data.columns),
                s3_path=f's3://{getenv("S3_BUCKET_NAME")}/{filename}',
                iam_role=getenv('AWS_S3_IAM_ROLE'),))
        self.connector.commit()
        return None


    def check_table_exists(self, table_name: str, schema:str='public') -> bool:
        self.cursor.execute(f"select exists(select * from information_schema.tables where table_name={schema + '.' + table_name})")
        return  self.cursor.fetchone()[0]


    def delete_table(self, table_name: str, schema:str='public'):
        self.cursor.execute(f"DROP TABLE IF EXISTS {schema +'.'+ table_name}")
        return None