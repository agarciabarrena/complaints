import nltk
import re
import pandas as pd


nltk.download('punkt')
nltk.download('stopwords')

'''
Manually download punkt (http://www.nltk.org/nltk_data/) and install it in:
OSX: /usr/local/share/nltk_data/tokenizers
Unix: /usr/share/nltk_data/tokenizers
'''
extra_stopwords = ['!', '¡', '?', '¿', '.', "'", '(', ')', ':', '´', ',', '&', '’', '$', '-', ';']

def text_preparation(df: pd.DataFrame, language: str='english', text_column: str='message'):
    df = df.copy()
    # We will start with the natural language processing
    # Tokenization:
    messages = df[text_column].copy()
    messages = messages.apply(lambda document: nltk.word_tokenize(str(document)))

    # Stop words removal:
    stop_words = nltk.corpus.stopwords.words(language)

    messages = messages.apply(lambda document:[term for term in document if term not in stop_words])
    messages = messages.apply(lambda document:[term for term in document if not re.match('[0-9]+[.]*',term)])  # rm numbers
    for pattern in extra_stopwords:
        messages = messages.apply(lambda document: [term for term in document if not re.match(f'^[{pattern}]*$', term)])

    # Word stemming
    snowball_stemmer = nltk.stem.SnowballStemmer(language)
    messages = messages.apply(lambda document: [snowball_stemmer.stem(term) for term in document])
    messages_not_empty_index = messages.loc[messages.str.len() != 0].index # List not empty

    df = df.loc[messages_not_empty_index].assign(processed_message=messages.loc[messages_not_empty_index])
    return df