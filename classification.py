import torch
from torch.autograd import Variable
import sys
import json

#You can write your own classification file to use the module
from attention.model import StructuredSelfAttentionForNLI
from attention.train import train,get_activation_wts,evaluate
from utils.data_loader import load_data_set
from visualization.attention_visualization import createHTML

import datetime, time
 
classified = False
classification_type = sys.argv[1]
if sys.argv[2]:
  mode = sys.argv[2] #train, test
else:
  mode = "train"

dtype = torch.cuda
 
def json_to_dict(json_set):
    for k,v in json_set.items():
        if v == 'False':
            json_set[k] = False
        elif v == 'True':
            json_set[k] = True
        else:
            json_set[k] = v
    return json_set
 
 
with open('config.json', 'r') as f:
    params_set = json.load(f)
 
with open('model_params.json', 'r') as f:
    model_params = json.load(f)
 
params_set = json_to_dict(params_set)
model_params = json_to_dict(model_params)
 
print("Using settings:",params_set)
print("Using model settings",model_params)
 
def visualize_attention(wts,x_test_pad,word_to_id,filename):
    wts_add = torch.sum(wts,1)
    wts_add_np = wts_add.data.cpu().numpy()
    wts_add_list = wts_add_np.tolist()
    id_to_word = {v:k for k,v in word_to_id.items()}
    text= []
    for test in x_test_pad:
        text.append(" ".join([id_to_word.get(i) for i in test]))
    createHTML(text, wts_add_list, filename)
    print("Attention visualization created for {} samples".format(len(x_test_pad)))
    return
 
def binary_classfication(attention_model,train_loader,epochs=5,use_regularization=True,C=1.0,clip=True):
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.RMSprop(attention_model.parameters())
    train(attention_model,train_loader,loss,optimizer,epochs,use_regularization,C,clip)
 
def multiclass_classification(attention_model,train_loader,epochs=5,use_regularization=True,C=1.0,clip=True):
    loss = torch.nn.NLLLoss()
    optimizer = torch.optim.RMSprop(attention_model.parameters())
    train(attention_model,train_loader,loss,optimizer,epochs,use_regularization,C,clip)
 
 
if __name__ == "__main__": 
  MAXLENGTH = model_params['timesteps']
  snli, TEXT, LABEL, GENRE = load_data_set("nli", MAXLENGTH, model_params['vocab_size'], model_params['batch_size'], dataset="snli")
  # train_loader,train_set,test_set,x_test_pad,word_to_id = load_data_set(1,MAXLENGTH,model_params["vocab_size"],model_params['batch_size']) #load the reuters dataset
  train_loader, val_loader, test_loader = snli
  attention_model = StructuredSelfAttentionForNLI(batch_size=train_loader.batch_size,
                                                  lstm_hid_dim=model_params['lstm_hidden_dimension'],
                                                  d_a = model_params["d_a"],
                                                  r=params_set["attention_hops"],
                                                  m_dim=20,
                                                  max_len=MAXLENGTH,
                                                  n_classes=3,
                                                  vocab=TEXT.vocab,
                                                  emb_dim=50).cuda()
  
  print("Model summary: ")
  print(attention_model)
  
  #Using regularization and gradient clipping at 0.5 (currently unparameterized)
  multiclass_classification(attention_model,train_loader,epochs=params_set["epochs"],use_regularization=params_set["use_regularization"],C=params_set["C"],clip=params_set["clip"])
  classified=True
  #wts = get_activation_wts(multiclass_attention_model,Variable(torch.from_numpy(x_test_pad[:]).type(dtype.LongTensor)))
  #print("Attention weights for the data in multiclass classification are:",wts)

  if classified:
      # test_last_idx = 1000
      # wts = get_activation_wts(attention_model,Variable(torch.from_numpy(x_test_pad[:test_last_idx]).type(dtype.LongTensor)).cuda())
      # print(wts.size())
      # visualize_attention(wts,x_test_pad[:test_last_idx],word_to_id,filename='attention.html')

      model_file = "models/model-{}".format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S'))
      print("Saving to {}...".format(model_file))
      torch.save(attention_model.state_dict(), model_file)
      print("Saved.")
