import torch
from torch import nn


# class SentimentClassifier(nn.Module):
#     def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(input_dim, embedding_dim)
#         self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, text):
#         embedded = self.embedding(text)
#         _, (hidden, _) = self.rnn(embedded)
#         return self.fc(hidden.squeeze(0))

class SentimentClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.rnn(embedded)
        if self.rnn.bidirectional:
            # Concatenate the last hidden state from the forward and backward directions
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden.squeeze(0)
        return self.fc(hidden)

class SentimentClassifier2(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim=1, num_layers=1, dropout=0, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.rnn(embedded)
        if self.rnn.bidirectional:
            # Concatenate the last hidden state from the forward and backward directions
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]  # Updated to handle unidirectional case
        return self.fc(hidden)
