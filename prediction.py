import torch
import torch.nn as nn
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from torchtext.data import get_tokenizer

from processing import preprocess_text
from sentiment_classifier import SentimentClassifier
from sentiment_classifier import SentimentClassifier2


# Load the vocab and the trained model
vocab = torch.load('models/vocab.pt', map_location=torch.device('cpu'))

# Initialize the model
input_dim = len(vocab)
embedding_dim = 100
hidden_dim = 128
output_dim = 1
num_layers = 3
dropout_rate = 0.5
model = SentimentClassifier2(input_dim, embedding_dim, hidden_dim, output_dim, num_layers, 0.5, False)
model.load_state_dict(torch.load('models/sentiment_model.pt', map_location=torch.device('cpu')))
model.eval()

# Load the tokenizer
nlp = spacy.load("en_core_web_sm")

# Tokenizer function
def tokenizer(text):
    return [tok.text for tok in nlp.tokenizer(text)]

# Text pipeline function
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]


# def predict_sentiment(text, model, vocab):
#     model.eval()
#     preprocessed_text = preprocess_text(text, remove_stop_words=True, stemming=False, lemmatization=True)
#     with torch.no_grad():
#         text_tokens = torch.tensor(text_pipeline(preprocessed_text)).unsqueeze(0)
#         output = model(text_tokens)
#         prediction_score = torch.sigmoid(output)
#         prediction = torch.round(prediction_score).item()
#     return prediction

def predict_sentiment(text, model, vocab):
    model.eval()
    preprocessed_text = preprocess_text(text, remove_stop_words=False, stemming=False, lemmatization=False)
    print("Original text:", text)
    print("Preprocessed text:", preprocessed_text)
    with torch.no_grad():
        text_tokens = torch.tensor(text_pipeline(preprocessed_text)).unsqueeze(0)
        output = model(text_tokens)
        prediction_score = torch.sigmoid(output).item()
    return prediction_score



def predict_sentiments_from_file(file_path, model, vocab):
    predictions = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            sentiment = predict_sentiment(line, model, vocab)
            predictions.append({'text': line, 'sentiment': sentiment})
    return pd.DataFrame(predictions)

def plot_sentiment_distribution(dataframe):
    plt.scatter(dataframe.index, dataframe['sentiment'], alpha=0.5)
    plt.xlabel('Index')
    plt.ylabel('Sentiment Value')
    plt.title('Sentiment Distribution')
    plt.show()

def plot_sentiment_counts(dataframe):
    sentiment_counts = dataframe['sentiment'].apply(lambda x: 'Positive' if x > 0.5 else 'Negative').value_counts()
    sentiment_counts.plot(kind='bar')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Counts')
    plt.show()

# Example usage
# example_text = "Last release was much better"
# sentiment = predict_sentiment(example_text, model, vocab)
# print(f"The sentiment for the text '{example_text}' is {sentiment}")


example_text = "it is really horrible"
sentiment_score = predict_sentiment(example_text, model, vocab)
print(f"The sentiment score for the text '{example_text}' is {sentiment_score}")


# Predict sentiments from a .txt file and plot the distribution
# input_file = 'input_text.txt'
# predictions_df = predict_sentiments_from_file(input_file, model, vocab)
# print(predictions_df.head())
# plot_sentiment_distribution(predictions_df)
# plot_sentiment_counts(predictions_df)