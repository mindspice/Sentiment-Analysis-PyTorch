import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import seaborn as sns
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import re

from processing import preprocess_text

dataset = 'data/sentiment140_short.csv'

df = pd.read_csv(dataset, encoding='ISO-8859-1')
df['text'] = df['text'].apply(lambda x: preprocess_text(x, remove_stop_words=True, stemming=False, lemmatization=True))
df['sentiment'] = df['sentiment'].apply(lambda x: 0 if x == 0 else 1)
df.to_csv('data/sentiment140_short_tokenized.csv', index=False)



# df['sentiment'].value_counts().sort_index().plot(kind='bar')
# plt.show()


