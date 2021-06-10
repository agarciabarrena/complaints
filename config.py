import logging
FORMAT = '%(asctime) %(user) %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger('complaints')

MODELS_FOLDER = 'models'
training_data_path = 'data_raw/training_data.csv'
TEXT_COL = 'message'
TRANSLATED_COL = 'message_translated'
LANGUAGE_COL = 'language'
RISK_SCORE_COL = 'risk_score'

