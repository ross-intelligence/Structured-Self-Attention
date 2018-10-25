import torch,keras
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data_utils


dtype = torch.cuda

class StructuredSelfAttentionForNLI(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup. 
    """
   
    def __init__(self,batch_size,lstm_hid_dim,d_a,r,max_len,emb_dim=100,vocab_size=None,type=0,n_classes = 1):
        """
        Initializes parameters suggested in paper
 
        Args:
            batch_size  : {int} batch_size used for training
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            max_len     : {int} number of lstm timesteps
            emb_dim     : {int} embeddings dimension
            vocab_size  : {int} size of the vocabulary
            use_pretrained_embeddings: {bool} use or train your own embeddings
            embeddings  : {torch.FloatTensor} loaded pretrained embeddings
            type        : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes
 
        Returns:
            self
 
        Raises:
            Exception
        """
        super(StructuredSelfAttention,self).__init__()
        import ipdb; ipdb.set_trace()
       
        self.embeddings,emb_dim = self._load_embeddings(use_pretrained_embeddings,embeddings,vocab_size,emb_dim)
        
        self.lstm = torch.nn.LSTM(emb_dim,lstm_hid_dim,1,batch_first=True)
        #                           50,       50
        self.linear_first = torch.nn.Linear(lstm_hid_dim,d_a)
        #                                          50    100 
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a,r)
        #                                    100,20
        self.linear_second.bias.data.fill_(0)

        # create 2 weight matrices wfh and wfp: batch_size x lstm_hid_dim x multiplicative_interaction_dim

        self.n_classes = n_classes
        self.linear_final = torch.nn.Linear(multiplicative_interaction_dim,self.n_classes)
        #                                        50,           1
        self.batch_size = batch_size       
        self.max_len = max_len
        self.lstm_hid_dim = lstm_hid_dim
        self.hidden_state = self.init_hidden()
        self.r = r
        self.type = type
                 
    def forward(self,x):
        embeddings = self.embeddings(x)       
        outputs, self.hidden_state = self.lstm(embeddings.view(self.batch_size,self.max_len,-1),self.hidden_state)       
        # (1024, 200, 50) <- (1024, 200, 50)
        x = torch.tanh(self.linear_first(outputs))
        # (1024, 200, 100) <- (1024, 200, 50)
        x = self.linear_second(x)       
        # (1024, 200, 20) <- (1024, 200, 100)
        x = self.softmax(x,1)       
        attention = x.transpose(1,2)       
        # (1024, 20, 200) <- (1024, 200, 20)
        sentence_embeddings = attention@outputs       
        # (1024, 20, 50) <- (1024, 20, 200)@(1024, 200, 50)
        avg_sentence_embeddings = torch.sum(sentence_embeddings,1)/self.r

        ## Do above for both h and p

        # Fh = bmm(avg_sentence_embeddings_h.view(n, 1, emb), self.wfh)
        # Fp = bmm(avg_sentence_embeddings_h.view(n, 1, emb), self.wfp)
       
        # interacted_embeddings = dot(Fh, Fp)
       
        if not bool(self.type):
            output = torch.sigmoid(self.linear_final(interacted_embeddings))
           
            return output,attention
        else:
            return F.log_softmax(self.linear_final(interacted_embeddings)),attention
       
	   
    def _load_embeddings(self,use_pretrained_embeddings,embeddings,vocab_size,emb_dim):
        """Load the embeddings based on flag"""
       
        if use_pretrained_embeddings is True and embeddings is None:
            raise Exception("Send a pretrained word embedding as an argument")
           
        if not use_pretrained_embeddings and vocab_size is None:
            raise Exception("Vocab size cannot be empty")
   
        if not use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(vocab_size,emb_dim,padding_idx=0)
            
        elif use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
            word_embeddings.weight = torch.nn.Parameter(embeddings)
            emb_dim = embeddings.size(1)
            
        return word_embeddings,emb_dim
       
        
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
