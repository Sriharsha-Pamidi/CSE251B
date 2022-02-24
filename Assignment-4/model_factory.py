################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

# Build and return the model here based on the configuration.

import torch
import torch.nn as nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import WeightedRandomSampler
from torch.autograd import Variable
import numpy as np
import pdb

def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    vocab_len=len(vocab)
    print("*************8vocab size ***************************   ",vocab_len)
    model_lstm = Img_Cap(model_type, embedding_size, hidden_size, vocab_len)
    
    return model_lstm
    
#     raise NotImplementedError("Model Factory Not Implemented")

class Img_Cap(nn.Module):
    
    def __init__(self, model_type, embedding_size, hidden_size, vocab_len):
        super().__init__()
        self.model_type = model_type
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_len=vocab_len
        self.encoder = ResNet50CNN(self.embedding_size)
        self.decoder = Decoder( self.embedding_size, self.hidden_size, self.vocab_len, self.model_type)

    def forward(self, images,captions):
        encoded_images = self.encoder(images)
        output_captions = self.decoder(encoded_images,captions)
        
        return output_captions
            
        

class ResNet50CNN(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.embedding_size = embedding_size
        
        self.pretrained_model = models.resnet50(pretrained = True)
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
            
        self.num_input_ftrs = self.pretrained_model.fc.in_features
        
        self.pretrained_model = nn.Sequential(*list(self.pretrained_model.children())[:-1])
       
        self.linear = nn.Linear(self.num_input_ftrs , self.embedding_size)
        #batch norm???
        
        
    def forward(self,images):
        

        features = self.pretrained_model(images)
        
        features = features.reshape(features.size(0),-1)
        encoded_image = self.linear(features)
        
        return encoded_image
        
    
    
class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_len, model_type):
        super().__init__()

        self.num_layers = 2
        self.embedding_size = embedding_size
        self.hidden_size    = hidden_size
        self.vocab_len      = vocab_len
        self.model_type     = model_type
        #use word2vec??
        
        self.embedding=nn.Embedding(self.vocab_len, self.embedding_size)
        
        #in baseline put no_layers=2
        if self.model_type == 'LSTM':
            self.layer = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True)
        elif self.model_type == 'RNN':
            self.layer = nn.RNN(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True)
            
        self.linear=nn.Linear(self.hidden_size, self.vocab_len)
                              
            
            
    
                              
    def forward(self, encoded_images, captions):
                       
        word_embeddings = self.embedding(captions)
        embeddings      = torch.cat((encoded_images.unsqueeze(1), word_embeddings),1)
        hidden,_        = self.layer(embeddings)
        vocab_output    = self.linear(hidden[0])
        
                              
        return vocab_output[1:]
                              
        

                              
        