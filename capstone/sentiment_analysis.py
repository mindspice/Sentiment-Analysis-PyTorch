import csv
import os
import torch
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from processing import PreProcessor
from sentiment_classifier import get_cnn_lstm_model


class SentimentDataset(Dataset):
    def __init__(self, vocab, csv_file, sample_fraction=1.0, random_seed=None):
        data = pd.read_csv(csv_file)
        self.nlp = spacy.load('en_core_web_lg')
        self.pre_processor = PreProcessor()
        self.vocab = vocab
        if 0 < sample_fraction < 1.0:
            data = data.sample(frac=sample_fraction, random_state=random_seed)
        self.data = data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentiment = torch.tensor(self.data.loc[idx, 'sentiment']).long()
        text = self.data.loc[idx, 'text']
        preprocessed_text = self.pre_processor.preprocess_text(text)
        text_tensor = torch.tensor(self.text_pipeline(preprocessed_text)).long()
        return sentiment, text_tensor

    def text_pipeline(self, text):
        return [self.vocab[token] for token in self.tokenizer(text)]

    def tokenizer(self, text):
        return [tok.text for tok in self.nlp.tokenizer(text)]


class SentimentAnalyzer:

    def __init__(self, vocab_path, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab = torch.load(vocab_path, map_location=self.device)
        self.model = self.load_model(model_path)
        self.nlp = spacy.load('en_core_web_lg')
        self.pre_processor = PreProcessor()
        self.model.to(self.device)

    def load_model(self, model_path):
        model = get_cnn_lstm_model(len(self.vocab))
        model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def tokenizer(self, text):
        return [tok.text for tok in self.nlp.tokenizer(text)]

    def text_pipeline(self, text):
        return [self.vocab[token] for token in self.tokenizer(text)]

    def predict_sentiment(self, text):
        self.model.eval()
        preprocessed_text = self.pre_processor.preprocess_text(text)
        with torch.no_grad():
            text_tokens = torch.tensor(self.text_pipeline(preprocessed_text)).unsqueeze(0).to(self.device)

            output = self.model(text_tokens)
            prediction_score = torch.sigmoid(output).item()
        return prediction_score

    def eval(self, test_set_file, sample_fraction=1.0, random_seed=None):
        if sample_fraction <= 0:
            return []
        test_dataset = SentimentDataset(self.vocab, test_set_file, sample_fraction=sample_fraction,
                                        random_seed=random_seed)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        accuracies = []

        correct = 0
        total = 0

        self.model.eval()

        with torch.no_grad():
            i = 1
            pred_sum = 0.0
            for sentiment, text in test_loader:
                sentiment = sentiment.to(self.device)
                text = text.to(self.device)
                output = self.model(text)
                prediction_score = torch.sigmoid(output).item()
                predicted = torch.tensor([1 if prediction_score >= 0.5 else 0]).to(self.device)
                total += sentiment.size(0)
                correct += (predicted == sentiment).sum().item()
                pred_sum += (predicted == sentiment).sum().item()
                if i % 10 == 0:
                    accuracies.append(pred_sum / 10)
                    pred_sum = 0
                i += 1

        accuracy = correct / total
        return accuracies, accuracy

    def predict_sentiments_from_file(self, file_path):
        predictions = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row['text']
                sentiment = self.predict_sentiment(text)
                predictions.append({'id': row['id'], 'text': text, 'sentiment': sentiment})
        return pd.DataFrame(predictions)

    @staticmethod
    def plot_sentiment_distribution(dataframe):
        plt.scatter(dataframe.index, dataframe['sentiment'], alpha=0.5)
        plt.xlabel('Index')
        plt.ylabel('Sentiment Value')
        plt.title('Sentiment Distribution')
        plt.show()

    @staticmethod
    def plot_sentiment_classifications(dataframe):
        sentiment_counts = dataframe['sentiment'].apply(
            lambda x: 'Positive' if x >= 0.67 else ('Neutral' if 0.34 <= x < 0.67 else 'Negative')
        ).value_counts()
        sentiment_counts.plot(kind='bar')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.title('Sentiment Classifications')
        plt.xticks(rotation=0)
        plt.show()

    @staticmethod
    def plot_statistics(dataframe):
        highest = dataframe['sentiment'].max()
        lowest = dataframe['sentiment'].min()
        average = dataframe['sentiment'].mean()

        # Prepare data for plotting
        labels = ['Highest', 'Average', 'Lowest']
        data = [highest, average, lowest]

        # Create bar plot
        plt.bar(labels, data)
        plt.ylabel('Value')
        plt.title('Highest, Lowest, and Average Sentiments')
        plt.show()

    @staticmethod
    def write_sentiments_to_csv(df, filename):
        output_directory = 'output/'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        df.to_csv(f'{output_directory}/{filename}_all.csv', index=False)
        neutral_df = df[(df['sentiment'] >= 0.34) & (df['sentiment'] <= 0.67)]
        neutral_df.to_csv(f'{output_directory}/{filename}_neutral.csv', index=False)
        negative_df = df[df['sentiment'] < 0.34]
        negative_df.to_csv(f'{output_directory}/{filename}_negative.csv', index=False)
        positive_df = df[df['sentiment'] >= 0.67]
        positive_df.to_csv(f'{output_directory}/{filename}_positive.csv', index=False)

    @staticmethod
    def plot_accuracies(accuracies):
        indices = range(len(accuracies))
        plt.scatter(indices, accuracies)
        plt.xlabel('Index / 10')
        plt.ylabel('Accuracy')
        plt.title('Evaluation Accuracy Distribution')
        plt.show()
