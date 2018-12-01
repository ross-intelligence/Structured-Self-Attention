import torch
from torch.autograd import Variable


dtype = torch.cuda

def train(attention_model,train_loader,criterion,optimizer,epochs = 5,use_regularization = False,C=0,clip=False):
    """
        Training code
 
        Args:
            attention_model : {object} model
            train_loader    : {DataLoader} training data loaded into a dataloader
            optimizer       :  optimizer
            criterion       :  loss function. Must be BCELoss for binary_classification and NLLLoss for multiclass
            epochs          : {int} number of epochs
            use_regularizer : {bool} use penalization or not
            C               : {int} penalization coeff
            clip            : {bool} use gradient clipping or not
       
        Returns:
            accuracy and losses of the model
 
      
        """
    losses = []
    accuracy = []
    for i in range(epochs):
        print("Running EPOCH",i+1)
        total_loss = 0
        n_batches = 0
        correct = 0
       
        for batch_idx,train in enumerate(train_loader):
            if (train.hypothesis[0].shape[0] != train_loader.batch_size):
              continue
            
            attention_model.hidden_state = attention_model.init_hidden()
            h,p,y = Variable(train.hypothesis[0]).cuda(),Variable(train.premise[0]), Variable(train.label).cuda()
            y_pred,att_h,att_p = attention_model(h, p)
           
            #penalization AAT - I
            if use_regularization:
                att_hT = att_h.transpose(1,2)
                identity_h = torch.eye(att_h.size(1))
                identity_h = Variable(identity_h.unsqueeze(0).expand(train_loader.batch_size,att_h.size(1),att_h.size(1))).cuda()

                att_pT = att_p.transpose(1,2)
                identity_p = torch.eye(att_p.size(1))
                identity_p = Variable(identity_p.unsqueeze(0).expand(train_loader.batch_size,att_p.size(1),att_p.size(1))).cuda()
                penal = (attention_model.l2_matrix_norm(att_h@att_hT - identity_h) + attention_model.l2_matrix_norm(att_p@att_pT - identity_p)) / 2
            
            if not bool(attention_model.type) :
                #binary classification
                #Adding a very small value to prevent BCELoss from outputting NaN's
                correct+=torch.eq(torch.round(y_pred.type(dtype.DoubleTensor).squeeze(1)),y).data.sum()
                if use_regularization:
                    try:
                        out = torch.clamp(y_pred.type(dtype.DoubleTensor).squeeze(1)+1e-8, 0., 1.)
                        loss = criterion(out,y) + C * penal/train_loader.batch_size
                       
                    except RuntimeError as e:
                        raise Exception("BCELoss gets nan values on regularization. Either remove regularization or add very small values")
                else:
                    loss = criterion(y_pred.type(dtype.DoubleTensor).squeeze(1),y)
                
            
            else:
                
                correct+=torch.eq(torch.max(y_pred,1)[1],y.type(dtype.LongTensor)).data.sum()
                if use_regularization:
                    loss = criterion(y_pred,y) + (C * penal/train_loader.batch_size).type(dtype.FloatTensor)
                else:
                    loss = criterion(y_pred,y)
               
 
            total_loss+=loss.data
            optimizer.zero_grad()
            loss.backward()
           
            #gradient clipping
            if clip:
                torch.nn.utils.clip_grad_norm_(attention_model.parameters(),0.5)
            optimizer.step()
            n_batches+=1
        print("avg_loss is",total_loss/n_batches)
        print("Accuracy of the model",correct.to(dtype=torch.float)/(n_batches*train_loader.batch_size))
        losses.append(total_loss/n_batches)
        accuracy.append(correct.to(dtype=torch.float)/(n_batches*train_loader.batch_size))
    return losses,accuracy
 
 
def evaluate(attention_model,x_test,y_test):
    """
        cv results
 
        Args:
            attention_model : {object} model
            x_test          : {nplist} x_test
            y_test          : {nplist} y_test
       
        Returns:
            cv-accuracy
 
      
    """
   
    attention_model.batch_size = x_test.shape[0]
    attention_model.hidden_state = attention_model.init_hidden()
    x_test_var = Variable(torch.from_numpy(x_test).type(dtype.LongTensor)).cuda()
    y_test_pred,_ = attention_model(x_test_var)
    if bool(attention_model.type):
        y_preds = torch.max(y_test_pred,1)[1]
        y_test_var = Variable(torch.from_numpy(y_test).type(dtype.LongTensor)).cuda()
       
    else:
        y_preds = torch.round(y_test_pred.type(dtype.DoubleTensor).squeeze(1))
        y_test_var = Variable(torch.from_numpy(y_test).type(dtype.DoubleTensor)).cuda()
       
    return torch.eq(y_preds,y_test_var).data.sum().to(torch.float)/x_test_var.size(0)
 
def get_activation_wts(attention_model,x):
    """
        Get r attention heads
 
        Args:
            attention_model : {object} model
            x               : {torch.Variable} input whose weights we want
       
        Returns:
            r different attention weights
 
      
    """
    attention_model.batch_size = x.size(0)
    attention_model.hidden_state = attention_model.init_hidden()
    _,wts = attention_model(x)
    return wts
