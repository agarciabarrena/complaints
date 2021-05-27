import logging
FORMAT = '%(asctime) %(user) %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger('complaints')

MODELS_FOLDER = 'models'