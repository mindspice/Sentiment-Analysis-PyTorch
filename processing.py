import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def preprocess_text(text, remove_stop_words=True, stemming=False, lemmatization=False):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove @handle mentions
    text = re.sub(r'@\w+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'\W', ' ', text).lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words

    if remove_stop_words:
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]
    # Perform stemming

    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    # Perform lemmatization

    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Recombine
    preprocessed_text = " ".join(tokens)

    return preprocessed_text






