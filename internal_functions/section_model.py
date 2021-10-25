import torch.nn as nn
import torch
import numpy as np
import spacy

class classifier(nn.Module):
    """Sectioning model"""
    # define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        # Constructor
        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # lstm layer
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

        # dense layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # activation function
        self.act = nn.Softmax(dim=1)


    def forward(self, text, text_lengths):
        # text = [batch size,sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]

        # packed sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [batch size, num layers * num directions,hid dim]
        # cell = [batch size, num layers * num directions,hid dim]

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)

        # Final activation function
        outputs = self.act(dense_outputs)

        return outputs

import torch.optim as optim

def optimizer_and_loss(model,device):
    """Defines the employed optimizer and loss function.
    INPUT: Model to train, selected device (GPU or CPU)
    OUTPUT: Model allocated into device, optimizer and criterion"""
    # define optimizer and loss
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    return model,optimizer,criterion


# define metric
def class_accuracy(preds, y):
    """Obtains the accuracy of the model
    INPUT: Model prediction, ground truth labels
    OUTPUT: Accuracy value"""
    # round predictions to the closest integer
    rounded_preds = torch.argmax(preds,axis=1)

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

    # push to cuda if available

def train(model, iterator, optimizer, criterion):
    """Trains the sectioning model.
    INPUT: Model to train, Data iterator, Optimizer, Criterion
    OUTPUT:Trained model, Accuracy, Loss"""
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # set the model in training phase
    model.train()

    for batch in iterator:
        # resets the gradients after every batch
        optimizer.zero_grad()

        # retrieve text and no. of words
        text, text_lengths = batch.text

        # convert to 1D tensor
        predictions = model(text, text_lengths).squeeze()

        # compute the loss
        loss = criterion(predictions, batch.label)

        # compute the binary accuracy
        acc = class_accuracy(predictions, batch.label)

        # backpropage the loss and compute the gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return model, epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    """Evaluates the given model.
    INPUT: Trained model, data iterator, criterion
    OUTPUT: Loss, accuracy"""
    # initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    # deactivating dropout layers
    model.eval()

    # deactivates autograd
    with torch.no_grad():
        for batch in iterator:
            # retrieve text and no. of words
            text, text_lengths = batch.text

            # convert to 1d tensor
            predictions = model(text, text_lengths).squeeze()

            # compute loss and accuracy
            loss = criterion(predictions, batch.label)
            acc = class_accuracy(predictions, batch.label)

            # keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def predict_sentence(model,vocab,sentence):
    """Predicts the section value of a given sentence
    INPUT: Trained model, Model vocab, Sentence to predict
    OUTPUT: Assigned section to the sentence"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nlp=spacy.load('en_core_sci_md')
    model=model.to(device)
    tokens=[t.text for t in nlp.tokenizer(sentence)]
    indexed = [vocab[t] for t in tokens]
    tensor_to_predict=torch.LongTensor(indexed).to(device)
    tensor_to_predict=tensor_to_predict.unsqueeze(1).T
    length_tensor= torch.LongTensor([len(indexed)]).to(device)
    prediction=model(tensor_to_predict,length_tensor)
    return prediction.argmax(1).item()