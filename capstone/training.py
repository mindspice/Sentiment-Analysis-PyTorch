import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataset import random_split, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from processing import PreProcessor
from sentiment_classifier import get_cnn_lstm_model

dataset = 'training_data/sentiment140_condensed.csv'


# Class to hold dataset

class SentimentDataset(Dataset):
    def __init__(self, csv_file, sample_fraction=1.0, random_seed=None):
        data = pd.read_csv(csv_file)
        if 0 < sample_fraction < 1.0:
            data = data.sample(frac=sample_fraction, random_state=random_seed)
        self.data = data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentiment = self.data.loc[idx, 'sentiment']
        text = self.data.loc[idx, 'text']
        return sentiment, text


## Load dataset
train_dataset = SentimentDataset(dataset)

# Split the dataset
train_ratio = 0.85
valid_ratio = 0.1
test_ratio = 0.05

train_len = int(train_ratio * len(train_dataset))
valid_len = int(valid_ratio * len(train_dataset))
test_len = len(train_dataset) - train_len - valid_len

train_data, valid_data, test_data = random_split(train_dataset, [train_len, valid_len, test_len],
                                                 generator=torch.Generator().manual_seed(19923))


# Extract data from the test_data SentimentDataset and convert it to a pandas.DataFrame
test_data_indices = list(test_data.indices)
test_data_sentiments = [train_dataset.data.loc[i, 'sentiment'] for i in test_data_indices]
test_data_texts = [train_dataset.data.loc[i, 'text'] for i in test_data_indices]
test_data_df = pd.DataFrame({'sentiment': test_data_sentiments, 'text': test_data_texts})

# Save the test data to a CSV file
test_data_df.to_csv('test/test_set.csv', index=False)


# Init tokenizer and function to yield tokens

tokenizer = get_tokenizer('spacy', language='en_core_web_lg')
pre_processor = PreProcessor()


def yield_tokens(data_iter):
    for _, text in data_iter:
        if isinstance(text, str):
            preprocessed_text = pre_processor.preprocess_text(text)
            tokens = tokenizer(preprocessed_text)
            yield tokens


# Init vocab
vocab = build_vocab_from_iterator(yield_tokens(train_data))
vocab.unk_index = vocab['<unk>']
vocab.pad_index = vocab['<pad>']


# Convert labels as data set uses 4 for positive and 0 for negative
def label_pipeline(label):
    if label == 4:
        return 1
    else:
        return 0

def text_pipeline(text_tokens):
    return [vocab[token] for token in text_tokens]


# Init training/validation batches
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


# Load training/validation batches
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False, collate_fn=collate_batch)

# init model
model = get_cnn_lstm_model(len(vocab))

# Load optimizer for training
optimizer = optim.AdamW(model.parameters(), lr=1e-4, amsgrad=True, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, cooldown=2, patience=5,threshold=2.5e-3, verbose=True)

criterion = nn.BCEWithLogitsLoss()


def binary_accuracy(preds, y):
    probs = torch.sigmoid(preds)
    threshold = 0.5
    correct = ((probs >= threshold) == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion, scheduler, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    model.to(device)


    for batch_idx, batch in enumerate(iterator):
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


# Training parameters and output
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 200

for epoch in range(0, epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, scheduler, device)
    valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)

    print(f'Epoch: {epoch + 1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:.2f}%')
    scheduler.step(valid_loss)

    if epoch % 10 == 0:
        torch.save(model.state_dict(), str('models/sentiment_model_' + str(epoch) + '.pt'))
        torch.save(vocab, str('models/vocab_' + str(epoch) + '.pt'))

torch.save(model.state_dict(), 'models/sentiment_model.pt')
torch.save(vocab, 'models/vocab.pt')
