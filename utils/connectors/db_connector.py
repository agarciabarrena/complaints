from os import getenv
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import boto3
from typing import List

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


    def copy_df_to_redshift(self, data: pd.DataFrame, table_name: str) -> None:
        filename = f'{datetime.now().strftime("%Y-%m-%d")}/{table_name}-{datetime.now().strftime("%H:%M:%S")}.csv'
        logger.info(f"Caching dataframe to S3 at: {getenv('S3_BUCKET_NAME')}/temp/{filename}")
        boto3.resource('s3',
                       aws_secret_access_key=getenv('AWS_SECRET_ACCESS_KEY'),
                       aws_access_key_id=getenv('AWS_ACCESS_KEY_ID'))\
            .Bucket(getenv('S3_BUCKET_NAME'),).put_object(
            Body=data.to_csv(index=False),
            ContentType='text/csv',
            Key='temp/'+filename,
        )
        logger.info("Uploading dataframe to redshift")
        self.cursor.execute(
            "COPY {table_name} (\"{columns}\") FROM '{s3_path}' IAM_ROLE '{iam_role}' CSV TRIMBLANKS TRUNCATECOLUMNS BLANKSASNULL EMPTYASNULL IGNOREHEADER 1".format(
                table_name=table_name,
                columns='\",\"'.join(data.columns),
                s3_path=f's3://{getenv("S3_BUCKET_NAME")}/temp/{filename}',
                iam_role=getenv('AWS_S3_IAM_ROLE'),))
        self.connector.commit()
        return None

    def insert_many(self, table, columns: list, params: List[list]):
        try:
            with self.connector.cursor() as cursor:
                query = 'INSERT INTO {} ("{}") VALUES %s'.format(table, '","'.join(columns))
                print(f"SQL-START: {query}")
                psycopg2.extras.execute_values(
                    cursor,
                    query,
                    params,
                )
                self.connector.commit()
                print('SQL-DONE - affected rows: {}'.format(cursor.rowcount))
        except Exception as e:
            self.connector.rollback()
            raise e


    def __check_table_exists(self, table_name: str, schema:str='public') -> bool:
        self.cursor.execute(f"select exists(select * from information_schema.tables where table_name={schema + '.' + table_name})")
        return  self.cursor.fetchone()[0]


    def __delete_table(self, table_name: str, schema:str='public'):
        self.cursor.execute(f"DROP TABLE IF EXISTS {schema +'.'+ table_name}")
        return None