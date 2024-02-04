from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils import data

from utils import load_data_link_prediction_DDI, load_data_link_prediction_PPI, load_data_link_prediction_DTI, load_data_link_prediction_GDI, Data_DDI, Data_PPI, Data_DTI, Data_GDI
from models import ResidualGCN, SkipGNN
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import copy
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=15,
                    help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch_size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=64,
                    help='Number of hidden units for encoding layer 1.')
parser.add_argument('--hidden2', type=int, default=16,
                    help='Number of hidden units for encoding layer 2.')
parser.add_argument('--hidden_decode1', type=int, default=512,
                    help='Number of hidden units for decoding layer 1.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--data_path', default='data/GDI/fold1',help='path to data fold')
parser.add_argument('--network_type', default='GDI',help='choose from DDI, PPI, DTI')
parser.add_argument('--input_type', default='one_hot',help='choose from one-hot, node2vec')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
# global variable

# Load data
if args.network_type == 'DDI':
    adj, adj2, features, idx_map = load_data_link_prediction_DDI(args.data_path, args.input_type)
    Data_class = Data_DDI
elif args.network_type == 'PPI':
    adj, adj2, features, idx_map = load_data_link_prediction_PPI(args.data_path, args.input_type)
    Data_class = Data_PPI
elif args.network_type == 'DTI':
    adj, adj2, features, idx_map = load_data_link_prediction_DTI(args.data_path, args.input_type)
    Data_class = Data_DTI  
elif args.network_type == 'GDI':
    adj, adj2, features, idx_map = load_data_link_prediction_GDI(args.data_path, args.input_type)
    Data_class = Data_GDI  
    
params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': 6,
              'drop_last' : True}

df_train = pd.read_csv(args.data_path + '/train.csv')
df_val = pd.read_csv(args.data_path + '/val.csv')
df_test = pd.read_csv(args.data_path + '/test.csv')

training_set = Data_class(idx_map, df_train.label.values, df_train)
train_loader = data.DataLoader(training_set, **params)

validation_set = Data_class(idx_map, df_val.label.values, df_val)
val_loader = data.DataLoader(validation_set, **params)

test_set = Data_class(idx_map, df_test.label.values, df_test)
test_loader = data.DataLoader(test_set, **params)

# Model and optimizer
# model = SkipGNN(nfeat=features.shape[1],
#             nhid1=args.hidden1,
#             nhid2=args.hidden2,
#             nhid_decode1 = args.hidden_decode1,
#             dropout=args.dropout)

model = ResidualGCN(nfeat=features.shape[1],
            nhid1=args.hidden1,
            nhid2=args.hidden2,
            nhid_decode1 = args.hidden_decode1,
            dropout=args.dropout)

# for plot
idxs=[]
for _,(idx1,idx2) in training_set:
    idxs.append([idx1,idx2])
for _,(idx1,idx2) in validation_set:
    idxs.append([idx1,idx2])
for _,(idx1,idx2) in test_set:
    idxs.append([idx1,idx2])
idxs=torch.tensor(idxs)
y=torch.zeros(len(features))
y.scatter_(0,idxs[:,1],1)



optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    # adj2 = adj2.cuda()

loss_fct = torch.nn.BCELoss()
m = torch.nn.Sigmoid()
    
max_auc = 0
model_max = copy.deepcopy(model)

def test(loader, model):
    model.eval()
    y_pred = []
    y_label = []
    
    for i, (label, inp) in enumerate(loader):
        if args.cuda:
            label = label.cuda()
            
        output, _ = model(features, adj, inp)

        n = torch.squeeze(m(output))
            
        loss = loss_fct(n, label.float())
    
        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + output.flatten().tolist()
        outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label, outputs), loss

loss_history = []

# Train model
t_total = time.time()
print('Start Training...')
for epoch in range(args.epochs):
    t = time.time()
    print ('-------- Epoch '+ str(epoch + 1)+' --------')
    y_pred_train = []
    y_label_train = []
    
    for i, (label, inp) in enumerate(train_loader):
        if args.cuda:
            label = label.cuda()
        #print(inp[0].shape)
        model.train()
        optimizer.zero_grad()
        output, _ = model(features, adj, inp)

        n = torch.squeeze(m(output))            
        loss_train = loss_fct(n, label.float())
        loss_history.append(loss_train.item())
        loss_train.backward()
        optimizer.step()
        
        label_ids = label.to('cpu').numpy()
        y_label_train = y_label_train + label_ids.flatten().tolist()
        y_pred_train = y_pred_train + output.flatten().tolist()    
        
        if i % 100 == 0:
            print('epoch: ' + str(epoch+1) +'/ iteration: ' + str(i+1) + '/ loss_train: ' + str(loss_train.cpu().detach().numpy()))
        
    roc_train = roc_auc_score(y_label_train, y_pred_train)
    
    # validation after each epoch    
    if not args.fastmode:
        roc_val, prc_val, f1_val, loss_val = test(val_loader, model)
        if roc_val > max_auc:
                model_max = copy.deepcopy(model)
                max_auc = roc_val
                #torch.save(model, path)    
        print('epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'auroc_train: {:.4f}'.format(roc_train),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'auroc_val: {:.4f}'.format(roc_val),
              'auprc_val: {:.4f}'.format(prc_val),
              'f1_val: {:.4f}'.format(f1_val),
              'time: {:.4f}s'.format(time.time() - t))   

        
plt.plot(loss_history)        
        
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
auroc_test, prc_test, f1_test, loss_test = test(test_loader, model_max)
print('loss_test: {:.4f}'.format(loss_test.item()),'auroc_test: {:.4f}'.format(auroc_test), 'auprc_test: {:.4f}'.format(prc_test), 'f1_test: {:.4f}'.format(f1_test))   



df_train = pd.read_csv(args.data_path + '/train.csv')
plot_set = Data_class(idx_map, df_train.label.values, df_train)
plot_loader = data.DataLoader(training_set, batch_size=len(training_set))
with torch.no_grad():
    for label,inp in plot_loader:
        embeddings = model_max.embed(features, adj, inp)
        break
print(f'embeddings shape: {embeddings.shape}')
torch.save(embeddings.cpu().numpy(),'embeddings.pt')
np.save('labels.npy',y)
exit()
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
reduced_x = TSNE().fit_transform(embeddings.cpu().numpy())
 
plt.figure(figsize=(8, 8))
# ax = plt.subplot(aspect='equal')
sc = plt.scatter(reduced_x[:,0], reduced_x[:,1],c=y)#,cmap='Spectral')#, lw=0, s=40)
# plt.xlim(-25, 25)
# plt.ylim(-25, 25)
plt.axis('off')
# ax.axis('tight')
plt.savefig('tsne-generated.png', dpi=120)



