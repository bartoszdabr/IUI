from googletrans import Translator
import pandas as pd
import numpy as np


def translate_text(X):
    translator = Translator()
    translated_X = [translator.translate(text, src='pl', dest='en').text for text in X]
    return np.array(translated_X)


df = pd.read_csv('dbdata.csv', encoding='utf-8')
labels = df['label'].values.astype("U")
X = df['sample'].values.astype("U")
X = translate_text(X)
labels = translate_text(labels)
pd.DataFrame({'text': X, 'label': labels}).to_csv('dbdata_eng.csv', index=True)
