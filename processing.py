import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from torchtext.data import get_tokenizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
spacy.cli.download("en_core_web_sm")


tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

def preprocess_text(text, remove_stop_words=True, stemming=False, lemmatization=False):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove @handle mentions
    text = re.sub(r'@\w+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'\W', ' ', text).lower()
    # # Tokenize
    # tokens = tokenizer(text)
    # text = " ".join(tokens)

    # Remove stop words
    if remove_stop_words:
        stop_words = set(stopwords.words("english"))
        text = " ".join([token for token in text.split() if token not in stop_words])

    # Perform stemming
    if stemming:
        stemmer = PorterStemmer()
        text = " ".join([stemmer.stem(token) for token in text.split()])

    # Perform lemmatization
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        text = " ".join([lemmatizer.lemmatize(token) for token in text.split()])

    return text







