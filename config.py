import logging
FORMAT = '%(asctime) %(user) %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger('complaints')

# Local config:
MODELS_FOLDER = 'models'
training_data_path = 'data_raw/training_data.csv'

# Dataframe config:
TEXT_COL = 'message'
TRANSLATED_COL = 'message_translated'
LANGUAGE_COL = 'language'
RISK_SCORE_COL = 'risk_score'

# S3 config:
BUCKET_NAME = 'complaints-check-subscriptions'

# Redshift config:
OUTPUT_REDSHIFT_TABLE = 'dev.customer_care_flow_events'

