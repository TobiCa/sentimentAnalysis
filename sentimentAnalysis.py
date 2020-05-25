import pandas as pd
import torch
import torch.nn.functional as F
from torchtext import data
import torch.nn as nn
import random
import torch.optim as optim
import sys
import pickle
import os
import time
import spacy

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        text = text.permute(1, 0)
        
        embedded = self.embedding(text)
        
        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
            
        return self.fc(cat)
        

class SentimentAnalysis():
    def __init__(self):
        self.SEED = 12
        torch.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_bigrams(self, x):
        n_grams = set(zip(*[x[i:] for i in range(2)]))
        for n_gram in n_grams:
            x.append(' '.join(n_gram))
        return x

    def categorical_accuracy(self, preds, y):
        max_preds = preds.argmax(dim = 1, keepdim = True)
        correct = max_preds.squeeze(1).eq(y)
        return correct.sum() / torch.FloatTensor([y.shape[0]])

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def predict_class(self, model, sentence, vocab, min_len = 4):
        nlp = spacy.load('en')
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        indexed = [vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(self.device)
        tensor = tensor.unsqueeze(1)
        preds = model(tensor)
        max_preds = preds.argmax(dim = 1)
        return max_preds.item()

    def get_model_params(self):
        model_params = {}
        labels = {}
        vocab = {}
        try:
            with open('saved/modelParams.pkl', 'rb') as f:
                model_params = pickle.load(f)
        except:
            print('Model params not found in modelParam.pkl. Did you train the model yet?')
    
        try:
            with open('saved/labels.pkl', 'rb') as f:
                labels = pickle.load(f)
        except:
            print('Data labels not found in labels.pkl. Did you train the model yet?')
        try:
            with open('saved/vocab.pkl', 'rb') as f:
                vocab = pickle.load(f)
        except:
            print('Vocabulary not foun din vocab.pkl. Did you train the model yet?')
        return model_params, labels, vocab

    # Data must exist in ./data dir
    def load_data(self):
        print('Loading data...')
        start_time = time.time()
        self.TEXT = data.Field(tokenize = 'spacy', preprocessing=self.generate_bigrams, lower=True)
        self.LABEL = data.LabelField()

        train_data_fields = [("textID", None),
              ("Tweet", self.TEXT),
              ("Selected_Tweet", self.TEXT),
              ("STANCE", self.LABEL),
            ]

        test_data_fields = [("textID", None),
              ("Tweet", self.TEXT),
              ("STANCE", self.LABEL),
            ]

        train_set = data.TabularDataset(
            path='data/train.csv', 
            format='csv',
            skip_header=True,
            fields=train_data_fields)

        test_set = data.TabularDataset(
            path='data/test.csv', 
            format='csv',
            skip_header=True,
            fields=test_data_fields)

        validation_set, test_set = test_set.split(random_state = random.seed(self.SEED))
        end_time = time.time()
        elapsed_time = end_time - start_time
    
        print(f'Data loaded in: {int(elapsed_time)}s')
        return train_set, validation_set, test_set


    def build_vocab(self, trainSet, MAX_VOCAB_SIZE = 25_000):

        print(f'Building vocabulary with max size at {MAX_VOCAB_SIZE} words')

        self.TEXT.build_vocab(train_set, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.twitter.27B.100d", 
                 unk_init = torch.Tensor.normal_)

        self.LABEL.build_vocab(trainSet)
        print(f'Vocabulary length: {len(self.TEXT.vocab)}')
        print(f'Labels: {self.LABEL.vocab.stoi}')
        with open('saved/labels.pkl', 'wb') as f:
            pickle.dump(self.LABEL.vocab.itos, f)
        with open('saved/vocab.pkl', 'wb') as f:
            pickle.dump(self.TEXT.vocab, f)


    def get_iterators(self, train_set, validation_set, test_set, sort_key, BATCH_SIZE = 64):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_set, validation_set, test_set),
            batch_size = BATCH_SIZE,
            device = self.device,
            sort_key=sort_key,
            sort_within_batch=False
            )
        return train_iterator, valid_iterator, test_iterator

    # Returns instance of model and the loss function
    def get_model(self):
        model_params = {
            'vocab_size': len(self.TEXT.vocab),
            'embedding_dim': 100,
            'n_filters': 100,
            'filter_sizes': [2,3,4],
            'output_dim': len(self.LABEL.vocab),
            'dropout': 0.5,
            'pad_idx': self.TEXT.vocab.stoi[self.TEXT.pad_token]
        }
        model = CNN(**model_params)
        with open('saved/modelParams.pkl', 'wb') as f:
            pickle.dump(model_params, f)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        model = model.to(self.device)
        criterion = criterion.to(self.device)

        return model, criterion, optimizer


    def train(self, model, iterator, optimizer, criterion):
    
        epoch_loss = 0
        epoch_acc = 0
    
        model.train()
    
        for batch in iterator:
            
            optimizer.zero_grad()

            predictions = model(batch.Tweet).squeeze(1)
            
            loss = criterion(predictions, batch.STANCE)
            
            acc = self.categorical_accuracy(predictions, batch.STANCE)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(self, model, iterator, criterion):
        
        epoch_loss = 0
        epoch_acc = 0
        
        model.eval()
        
        with torch.no_grad():
        
            for batch in iterator:

                predictions = model(batch.Tweet)
                
                loss = criterion(predictions, batch.STANCE)
                
                acc = self.categorical_accuracy(predictions, batch.STANCE)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)





if (len(sys.argv) > 1 and sys.argv[1] == 'train'):
    sentimentAnalysis = SentimentAnalysis()
    train_set, validation_set, test_set = sentimentAnalysis.load_data()
    sentimentAnalysis.build_vocab(test_set)

    sort_key = lambda x:len(x.Tweet)

    train_iterator, valid_iterator, test_iterator = sentimentAnalysis.get_iterators(train_set, validation_set, test_set, sort_key, 124)

    model, criterion, optimizer = sentimentAnalysis.get_model()

    N_EPOCHS = 8

    best_valid_loss = float('inf')

    print(f'Training model with {N_EPOCHS} epochs...')
    for epoch in range(N_EPOCHS):
            
        start_time = time.time()
        
        train_loss, train_acc = sentimentAnalysis.train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = sentimentAnalysis.evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()

        epoch_mins, epoch_secs = sentimentAnalysis.epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model/sentimentModel.pt')
            print('Saved new model')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    print('Testing model on test data...')

    model.load_state_dict(torch.load('model/sentimentModel.pt'))

    test_loss, test_acc = sentimentAnalysis.evaluate(model, test_iterator, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

if (len(sys.argv) > 1 and sys.argv[1] == 'predict'):
    if (sys.argv[2] and len(sys.argv[2])):
        sentimentAnalysis = SentimentAnalysis()
        model_params, labels, vocab = sentimentAnalysis.get_model_params()

        if (len(model_params) and len(labels)):
            model = CNN(**model_params)
            model.load_state_dict(torch.load('model/sentimentModel.pt'))
            model.eval()
            pred_class = sentimentAnalysis.predict_class(model, sys.argv[2], vocab)
            print(f'The predicted sentiment is: {labels[pred_class]}')
        else:
            print('Could not find model parameters or labels in current directory. Did you train a model before trying this?')
    else:
        print('Oops, seems like you did not pass a sentence for prediction!')
