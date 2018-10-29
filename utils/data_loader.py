#Please create your own dataloader for new datasets of the following type

import torch
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import torch.utils.data as data_utils

from nli.data_loader import NLIDataloader

dtype = torch.cuda
 
def load_data_set(datatype,max_len,vocab_size,batch_size, dataset):
    nliLoader = NLIDataloader('./multinli_1.0', './snli_1.0', 'glove.6B.50d')
    data, TEXT, LABEL, GENRE = nliLoader.load_nlidata(batch_size, "cuda:0", "spacy", dataset=dataset, max_len=max_len)
    return data, TEXT, LABEL, GENRE
