import nltk
import re
import pandas as pd
from google.cloud import translate_v2 as translate
from config import TRANSLATED_COL, TEXT_COL, LANGUAGE_COL


nltk.download('punkt')
nltk.download('stopwords')

'''
Manually download punkt (http://www.nltk.org/nltk_data/) and install it in:
OSX: /usr/local/share/nltk_data/tokenizers
Unix: /usr/share/nltk_data/tokenizers
'''
extra_stopwords = ['!', '¡', '?', '¿', '.', "'", '(', ')', ':', '´', ',', '&', '’', '$', '-', ';', '+', '#', '\'']
language_dict = {'en': 'english'}

def text_preparation(df: pd.DataFrame,
                     language: str='en',
                     text_column: str=TEXT_COL,
                     translated_text_col: str=TRANSLATED_COL):
    df = df.copy()
    # We will start with the natural language processing

    # Clean empty messages and apply lower()
    df = df[df[text_column] != ""]  # Clean empty
    df = df.assign(message=df[text_column].str.lower())
    # Tokenization:
    messages = df[text_column].copy()
    messages = messages.apply(lambda document: nltk.word_tokenize(str(document)))

    # Remove punctuation marks:
    for pattern in extra_stopwords:
        messages = messages.apply(lambda document: [term for term in document if not re.match(f'^[{pattern}]*$', term)])
    df[text_column] = messages.str.join(' ')

    # Translate
    tr = Translator()
    df = tr.translate_batch(df)

    # Stop words removal:
    messages = df[translated_text_col].copy()
    language = language_dict.get(language)
    stop_words = nltk.corpus.stopwords.words(language)
    messages = messages.apply(lambda document: nltk.word_tokenize(str(document)))
    messages = messages.apply(lambda document:[term for term in document if term not in stop_words])
    messages = messages.apply(lambda document:[term for term in document if not re.match('[0-9]+[.]*',term)])  # rm numbers

    # Word stemming
    snowball_stemmer = nltk.stem.SnowballStemmer(language)
    messages = messages.apply(lambda document: [snowball_stemmer.stem(term) for term in document])
    messages_not_empty_index = messages.loc[messages.str.len() != 0].index # List not empty

    df = df.loc[messages_not_empty_index].assign(processed_message=messages.loc[messages_not_empty_index])
    return df

class Translator():
    def __init__(self):
        self.tr_client = translate.Client()

    def translate(self, message: str, output_language: str='en') -> str:
        if len(message) <= 1:  # Just one character
            return message
        else:
            output = self.tr_client.translate(message, target_language=output_language)
            translated_text = output["translatedText"]
            return translated_text

    def translate_batch(self, df: pd.DataFrame,
                        text_lang_column: str=LANGUAGE_COL,
                        text_column: str=TEXT_COL,
                        output_language: str='en',
                        output_translated_column: str=TRANSLATED_COL) -> pd.DataFrame:
        messages_diff_lang = df[df[text_lang_column] != output_language][text_column]
        for idx, msg in messages_diff_lang.items():
            messages_diff_lang.loc[idx] = self.translate(msg, output_language=output_language)
        df[output_translated_column] = df[text_column].copy()
        df.loc[messages_diff_lang.index, output_translated_column] = messages_diff_lang
        return df

