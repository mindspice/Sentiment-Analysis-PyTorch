import math

import torch
from torch import nn
import torch.nn.functional as F


# Model class with parameters
class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim=1, num_layers=1, dropout=0.0,
                 bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout,
                           bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        return self.fc(hidden)

class SentimentClassifierWithCNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim=1, num_layers=1, dropout=0.0,
                 bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.conv1d = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1)  # Added CNN layer
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout,
                           bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)
        conv_out = self.conv1d(embedded)
        conv_out = F.relu(conv_out)
        conv_out = conv_out.permute(0, 2, 1)
        _, (hidden, _) = self.rnn(conv_out)
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        return self.fc(hidden)


def get_lstm_model(vocab_size):
    input_dim = vocab_size
    embedding_dim = 200
    hidden_dim = 256
    output_dim = 1
    num_layers = 4
    dropout_rate = 0.25
    bidirectional = True
    return SentimentClassifier(input_dim, embedding_dim, hidden_dim, output_dim, num_layers, dropout_rate,
                               bidirectional)

def get_cnn_lstm_model(vocab_size):
    input_dim = vocab_size
    embedding_dim = 200
    hidden_dim = 256
    output_dim = 1
    num_layers = 4
    dropout_rate = 0.25
    bidirectional = True
    return SentimentClassifierWithCNN(input_dim, embedding_dim, hidden_dim, output_dim, num_layers, dropout_rate,
                               bidirectional)
