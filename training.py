import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import spacy

from processing import preprocess_text
from sentiment_classifier import SentimentClassifier
from sentiment_classifier import SentimentClassifier2

dataset = 'data/sentiment140_condensed.csv'


# Load full dataframe
# df = pd.read_csv(dataset)

# python -m spacy download en_core_web_sm

class SentimentDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data = self.data.sample(frac=0.1, random_state=42).reset_index(drop=True)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentiment = self.data.loc[idx, 'sentiment']
        text = self.data.loc[idx, 'text']
        return sentiment, text


# Create training and validation dataframes
train_dataset = SentimentDataset(dataset)
train_data, valid_data = torch.utils.data.random_split(
    train_dataset,
    [int(0.8 * len(train_dataset)),
     len(train_dataset) - int(0.8 * len(train_dataset))],
    generator=torch.Generator().manual_seed(42)
)

train_dataset = SentimentDataset(dataset)
train_data, valid_data = random_split(train_dataset, [int(0.8 * len(train_dataset)),
                                                      len(train_dataset) - int(0.8 * len(train_dataset))],
                                      generator=torch.Generator().manual_seed(42))

tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


def yield_tokens(data_iter):
    for _, text in data_iter:
        if isinstance(text, str):
            preprocessed_text = preprocess_text(text, remove_stop_words=False, stemming=False, lemmatization=False)
            tokens = tokenizer(preprocessed_text)
            yield tokens

vocab = build_vocab_from_iterator(yield_tokens(train_data))
vocab.unk_index = vocab['<unk>']

def label_pipeline(label):
    if label == 4:  # Assuming positive sentiment is represented as '4' in your dataset
        return 1
    else:
        return 0
def text_pipeline(text_tokens):
    return [vocab[token] for token in text_tokens]


def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        if isinstance(_text, float):
            _text = ''
        text_tokens = tokenizer(_text)
        text_list.append(torch.tensor(text_pipeline(text_tokens), dtype=torch.int64))
    labels = torch.tensor(label_list, dtype=torch.int64)
    texts = pad_sequence(text_list, padding_value=vocab['<pad>'], batch_first=True)
    return labels, texts



train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(valid_data, batch_size=64, shuffle=False, collate_fn=collate_batch)




# input_dim = len(vocab)
# embedding_dim = 100
# hidden_dim = 128
# output_dim = 1

input_dim = len(vocab)
embedding_dim = 100
hidden_dim = 128
output_dim = 1
num_layers = 3
dropout_rate = 0.5

#model = SentimentClassifier(input_dim, embedding_dim, hidden_dim, output_dim)

model = SentimentClassifier2(input_dim, embedding_dim, hidden_dim, output_dim, num_layers, dropout_rate, False)


optimizer = optim.AdamW(model.parameters(), lr=1e-5, amsgrad=True, weight_decay=5e-5)
#scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
criterion = nn.BCEWithLogitsLoss()


# def binary_accuracy(preds, y):
#     rounded_preds = torch.round(torch.sigmoid(preds))
#     correct = (rounded_preds == y).float()
#     acc = correct.sum() / len(correct)
#     return acc


def binary_accuracy(preds, y):
    # Calculate the probabilities without rounding
    probs = torch.sigmoid(preds)
    # Set a threshold to classify the output
    threshold = 0.5
    # Calculate the number of correct predictions
    correct = ((probs >= threshold) == y).float()
    # Calculate the accuracy
    acc = correct.sum() / len(correct)
    return acc

# def train(model, iterator, optimizer, criterion, device):
#     epoch_loss = 0
#     epoch_acc = 0
#     model.train()
#     model.to(device)
#
#     for batch in train_loader:
#         labels, texts = batch
#         labels, texts = labels.to(device), texts.to(device)
#         optimizer.zero_grad()
#         predictions = model(texts).squeeze(1)
#         loss = criterion(predictions, labels.float())
#         acc = binary_accuracy(predictions, labels)
#         loss.backward()
#         optimizer.step()
#         #scheduler.step()
#
#         epoch_loss += loss.item()
#         epoch_acc += acc.item()
#
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    model.to(device)

    for batch in train_loader:
        labels, texts = batch
        labels, texts = labels.to(device), texts.to(device)
        optimizer.zero_grad()
        predictions = model(texts).squeeze(1)
        loss = criterion(predictions, labels.float())
        acc = binary_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# def evaluate(model, iterator, criterion, device):
#     epoch_loss = 0
#     epoch_acc = 0
#     model.eval()
#     model.to(device)
#
#     with torch.no_grad():
#         for batch in iterator:
#             labels, texts = batch
#             labels, texts = labels.to(device), texts.to(device)
#
#             predictions = model(texts).squeeze(1)
#             loss = criterion(predictions, labels.float())
#             acc = binary_accuracy(predictions, labels)
#
#             epoch_loss += loss.item()
#             epoch_acc += acc.item()
#
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in iterator:
            labels, texts = batch
            labels, texts = labels.to(device), texts.to(device)

            predictions = model(texts).squeeze(1)
            loss = criterion(predictions, labels.float())
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 100

for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)

    print(f'Epoch: {epoch + 1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:.2f}%')

# for epoch in range(epochs):
#     train_loss, train_acc = train(model, train_loader, optimizer, scheduler, criterion, device)
#     valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
#
#     print(f'Epoch: {epoch + 1:02}')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
#     print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:.2f}%')
    if epoch % 10 == 0:
        torch.save(model.state_dict(), str('models/sentiment_model' + str(epoch) + '.pt'))
        torch.save(vocab, str('models/vocab_' + str(epoch) + '.pt'))

torch.save(model.state_dict(), 'models/sentiment_model.pt')
torch.save(vocab, 'models/vocab.pt')
