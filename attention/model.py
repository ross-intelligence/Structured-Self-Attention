import torch,keras
import numpy as np
import math
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.utils.data as data_utils


dtype = torch.cuda

class StructuredSelfAttentionForNLI(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup. 
    """
   
    def __init__(self,batch_size,lstm_hid_dim,d_a,r,m_dim,max_len,vocab=None,emb_dim=100,n_classes = 1):
        """
        Initializes parameters suggested in paper
 
        Args:
            batch_size  : {int} batch_size used for training
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            m_dim       : {int} size of the multiplicative interaction weight matrix
            max_len     : {int} number of lstm timesteps
            emb_dim     : {int} embeddings dimension
            use_pretrained_embeddings: {bool} use or train your own embeddings
            embeddings  : {torch.FloatTensor} loaded pretrained embeddings
            n_classes   : {int} number of classes
 
        Returns:
            self
 
        Raises:
            Exception
        """
        super(StructuredSelfAttentionForNLI,self).__init__()

        self.embeddings = torch.nn.Embedding.from_pretrained(vocab.vectors,sparse=True)

        self.lstm = torch.nn.LSTM(emb_dim,lstm_hid_dim,1,batch_first=True)
        #                           50,       50
        self.linear_first = torch.nn.Linear(lstm_hid_dim,d_a)
        #                                          50    100 
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a,r)
        #                                    100,20
        self.linear_second.bias.data.fill_(0)

        # create 2 weight matrices wfh and wfp: batch_size x lstm_hid_dim x m_dim 
        self.wfh = Parameter(torch.Tensor(batch_size, lstm_hid_dim, m_dim))
        stdv = 1. / math.sqrt(self.wfh.size(1))
        self.wfh.data.uniform_(-stdv, stdv)

        self.wfp = Parameter(torch.Tensor(batch_size, lstm_hid_dim, m_dim))
        stdv = 1. / math.sqrt(self.wfp.size(1))
        self.wfp.data.uniform_(-stdv, stdv)

        self.n_classes = n_classes
        self.linear_final = torch.nn.Linear(m_dim,self.n_classes)
        #                                        50,           1
        self.batch_size = batch_size       
        self.max_len = max_len
        self.lstm_hid_dim = lstm_hid_dim
        self.emb_dim = emb_dim
        self.hidden_state = self.init_hidden()
        self.r = r
                 
    def forward(self,h, p):
        h = self.embeddings(h)
        outputs_h, self.hidden_state = self.lstm(h.view(self.batch_size,self.max_len,-1),self.hidden_state)       
        # (1024, 200, 50) <- (1024, 200, 50)
        h = torch.tanh(self.linear_first(outputs_h))
        # (1024, 200, 100) <- (1024, 200, 50)
        h = self.linear_second(h)       
        # (1024, 200, 20) <- (1024, 200, 100)
        h = self.softmax(h,1)       
        attention_h = h.transpose(1,2)       
        # (1024, 20, 200) <- (1024, 200, 20)
        sentence_embeddings_h = attention_h@outputs_h
        # (1024, 20, 50) <- (1024, 20, 200)@(1024, 200, 50)
        avg_sentence_embeddings_h = torch.sum(sentence_embeddings_h,1)/self.r

        p = self.embeddings(p)
        outputs_p, self.hidden_state = self.lstm(p.view(self.batch_size,self.max_len,-1),self.hidden_state)       
        # (1024, 200, 50) <- (1024, 200, 50)
        p = torch.tanh(self.linear_first(outputs_p))
        # (1024, 200, 100) <- (1024, 200, 50)
        p = self.linear_second(p)       
        # (1024, 200, 20) <- (1024, 200, 100)
        p = self.softmax(p,1)       
        attention_p = p.transpose(1,2)       
        # (1024, 20, 200) <- (1024, 200, 20)
        sentence_embeddings_p = attention_p@outputs_p
        # (1024, 20, 50) <- (1024, 20, 200)@(1024, 200, 50)
        avg_sentence_embeddings_p = torch.sum(sentence_embeddings_p,1)/self.r
        
        Fh = torch.bmm(avg_sentence_embeddings_h.view(self.batch_size, 1, self.emb_dim), self.wfh).squeeze()
        Fp = torch.bmm(avg_sentence_embeddings_p.view(self.batch_size, 1, self.emb_dim), self.wfp).squeeze()
       
        interacted_embeddings = torch.mul(Fh, Fp)
       
        return F.log_softmax(self.linear_final(interacted_embeddings)),attention_p,attention_h
       
	   
    def softmax(self,input, axis=1):
        """
        Softmax applied to axis=n
 
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied
 
        Returns:
            softmaxed tensors
 
       
        """
 
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d, 0)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)
       
        
    def init_hidden(self):
        return (Variable(torch.zeros(1,self.batch_size,self.lstm_hid_dim)).cuda(),Variable(torch.zeros(1,self.batch_size,self.lstm_hid_dim)).cuda())
       
        
	#Regularization
    def l2_matrix_norm(self,m):
        """
        Frobenius norm calculation
 
        Args:
           m: {Variable} ||AAT - I||
 
        Returns:
            regularized value
 
       
        """
        return torch.sum(torch.sum(torch.sum(m**2,1),1)**0.5).type(dtype.DoubleTensor)
