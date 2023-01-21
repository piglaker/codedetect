import os
import re
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import pandas as pd
import matplotlib.pyplot as plt

torch.manual_seed(1)

class CNN(nn.Module):
    def __init__(self, input_size, embedding_dim, kernel_wins, num_class, embedding):
        super(CNN, self).__init__()
        if embedding is None:
            self.embedding = nn.Embedding(21168, embedding_dim, max_norm=True, padding_idx=-100)
        else:
            self.embedding = embedding
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, embedding_dim, size) for size in kernel_wins])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(len(kernel_wins)*embedding_dim, num_class)

    def forward(self, input_ids, labels, attention_mask):
        embeded_x = self.embedding(input_ids)

        embeded_x = embeded_x.transpose(1, 2)
        con_x = [conv(embeded_x) for conv in self.convs]

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        
        fc_x = torch.cat(pool_x, dim=1)

        fc_x = self.dropout(fc_x)

        fc_x = fc_x.squeeze(-1)

        logits = self.fc(fc_x)
        
        loss_function = torch.nn.CrossEntropyLoss()#BCEWithLogitsLoss()
        #print(logits.shape, labels.shape) 
        loss = loss_function( logits.float(), labels.long().squeeze(1) )
         
        return loss, logits
