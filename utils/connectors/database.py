import os
from datetime import datetime
from typing import List, Optional

import boto3
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from config import BUCKET_NAME


BUCKET_NAME_DB = BUCKET_NAME + '/temp'


def __execute(cursor, query, params=None, commit=False, row_count: int = None):

    if query[:8] == 'TRUNCATE':
        cursor.execute(f"SELECT count(*) FROM {query.split(' ')[1]}")
        row_count = cursor.fetchone()[0]

    print(f"SQL-START: {query}, params: {params}")
    cursor.execute(query, params)
    cursor.connection.commit() if commit else None

    print('SQL-DONE - affected rows: {}'.format(
        row_count if row_count is not None else cursor.rowcount))


def get_connection():
    return psycopg2.connect(os.getenv('CONNECTION_STRING'))


def insert_many(table, columns: list, params: List[list]):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            query = 'INSERT INTO {} ("{}") VALUES %s'.format(table, '","'.join(columns))
            print(f"SQL-START: {query}")
            psycopg2.extras.execute_values(
                cursor,
                query,
                params,
            )
            conn.commit()
            print('SQL-DONE - affected rows: {}'.format(cursor.rowcount))
    except Exception as e:
        conn.rollback()
        raise e


def copy_to_redshift(data: pd.DataFrame, table_name: str, truncate=False):
    filename = f'{datetime.now().strftime("%Y-%m-%d")}/{table_name}-{datetime.now().strftime("%H:%M:%S")}.csv'
    boto3.resource('s3').Bucket(BUCKET_NAME_DB).put_object(
        Body=data.to_csv(index=False),
        ContentType='text/csv',
        Key=filename,
    )

    conn = get_connection()
    with conn.cursor() as cursor:
        if truncate:
            __execute(cursor, f'TRUNCATE {table_name}')

        __execute(
            cursor,
            "COPY {table_name} (\"{columns}\") "
            "FROM '{s3_path}' "
            "IAM_ROLE '{iam_role}' "
            "CSV TRIMBLANKS TRUNCATECOLUMNS BLANKSASNULL EMPTYASNULL IGNOREHEADER 1".format(
                table_name=table_name,
                columns='\",\"'.join(data.columns),
                s3_path=f's3://{BUCKET_NAME_DB}/{filename}',
                iam_role=os.getenv('AWS_S3_IAM_ROLE'),
            ),
            params=None,
            commit=True,
            row_count=len(data)
        )


def select(query, params=None, connection=None) -> pd.DataFrame:
    conn = get_connection() if connection is None else connection
    with conn.cursor() as cursor:
        __execute(cursor, query, params)
        return pd.DataFrame(
            cursor.fetchall(),
            columns=[desc[0] for desc in cursor.description]
        )


def select_one(query, params=None) -> Optional[dict]:
    conn = get_connection()
    with conn.cursor() as cursor:
        __execute(cursor, query, params)
        result = cursor.fetchone()
        return dict(zip([desc[0] for desc in cursor.description], result)) if result else None


def delete(query, params):
    __execute(get_connection().cursor(), query, params, commit=True)
