import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np
import h5py as h5
import sys
from torchinfo import summary
from datetime import datetime

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gain + self.bias

class ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        
        if in_size != out_size:
            self.skip = nn.Linear(in_size, out_size, bias=False)
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()
        self.norm2 = Affine()

        self.act1 = nn.Tanh()#nn.PReLU()
        self.act2 = nn.Tanh()#nn.PReLU()

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.layer1(self.act1(self.norm1(x))) / np.sqrt(10)
        o2 = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10) + xskip

        return o2

class ResBottle(nn.Module):
    def __init__(self, size, N):
        super(ResBottle, self).__init__()

        self.size = size
        self.N = N
        encoded_size = size // N

        self.norm1  = torch.nn.BatchNorm1d(encoded_size)
        self.layer1 = nn.Linear(size,encoded_size)
        self.act1   = nn.Tanh()

        self.norm2  = torch.nn.BatchNorm1d(encoded_size)
        self.layer2 = nn.Linear(encoded_size,encoded_size)
        self.act2   = nn.Tanh()

        self.norm3  = torch.nn.BatchNorm1d(size)
        self.layer3 = nn.Linear(encoded_size,size)
        self.act3   = nn.Tanh()

        self.skip     = nn.Identity()#nn.Linear(size,size)
        self.act_skip = nn.Tanh()

    def forward(self, x):
        x_skip = self.act_skip(self.skip(x))

        o1 = self.act1(self.norm1(self.layer1(x)/np.sqrt(10)))
        o2 = self.act2(self.norm2(self.layer2(o1)/np.sqrt(10)))
        o3 = self.norm3(self.layer3(o2))
        o  = self.act3(o3+x_skip)

        return o

class DenseBlock(nn.Module):
    def __init__(self, size):
        super(DenseBlock, self).__init__()

        self.skip = nn.Identity()

        self.layer1 = nn.Linear(size, size)
        self.layer2 = nn.Linear(size, size)

        self.norm1 = torch.nn.BatchNorm1d(size)
        self.norm2 = torch.nn.BatchNorm1d(size)

        self.act1 = nn.Tanh()#nn.SiLU()#nn.PReLU()
        self.act2 = nn.Tanh()#nn.SiLU()#nn.PReLU()

    def forward(self, x):
        xskip = self.skip(x)
        o1    = self.layer1(self.act1(self.norm1(x))) / np.sqrt(10)
        o2    = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10)
        o     = torch.cat((o2,xskip),axis=1)
        return o

class Attention(nn.Module):
    def __init__(self, in_size ,n_partitions):
        super(Attention, self).__init__()
        self.embed_dim    = in_size//n_partitions
        self.WQ           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WK           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WV           = nn.Linear(self.embed_dim,self.embed_dim)
        self.act          = nn.Softmax(dim=2)
        self.scale        = np.sqrt(self.embed_dim)
        self.n_partitions = n_partitions
        self.norm         = torch.nn.BatchNorm1d(in_size)

    def forward(self, x):
        batchsize = x.shape[0]
        x_norm    = self.norm(x)

        Q = torch.empty((batchsize,self.embed_dim,self.n_partitions))
        K = torch.empty((batchsize,self.embed_dim,self.n_partitions))
        V = torch.empty((batchsize,self.embed_dim,self.n_partitions))

        # stack the input to find Q,K,V
        for i in range(self.n_partitions):
            qi = self.WQ(x_norm[:,i*self.embed_dim:(i+1)*self.embed_dim])
            ki = self.WK(x_norm[:,i*self.embed_dim:(i+1)*self.embed_dim])
            vi = self.WV(x_norm[:,i*self.embed_dim:(i+1)*self.embed_dim])

            Q[:,:,i] = qi
            K[:,:,i] = ki
            V[:,:,i] = vi

        # compute weighted dot product
        dot_product = torch.bmm(Q,K.transpose(1, 2).contiguous())
        normed_mat  = self.act(dot_product/self.scale)
        prod        = torch.bmm(normed_mat,V)

        #concat results of each head
        out = torch.cat(tuple([prod[:,i] for i in range(self.embed_dim)]),dim=1)+x

        return out

class Transformer(nn.Module):
    def __init__(self, n_heads, int_dim):
        super(Transformer, self).__init__()  
    
        self.int_dim     = int_dim
        self.n_heads     = n_heads
        self.module_list = nn.ModuleList([nn.Linear(int_dim,int_dim) for i in range(n_heads)])
        self.act         = nn.Tanh()#nn.SiLU()
        self.norm        = torch.nn.BatchNorm1d(int_dim*n_heads)

    def forward(self,x):
        # init result array
        batchsize = x.shape[0]
        x_norm = self.norm(x)
        results = torch.empty((batchsize,self.int_dim,self.n_heads))

        # do mlp for each head
        for i,layer in enumerate(self.module_list):
            o = x_norm[:,i*self.int_dim:(i+1)*self.int_dim]
            o = self.act(layer(o))
            results[:,:,i] = o

        # concat heads
        out = torch.cat(tuple([results[:,i] for i in range(self.int_dim)]),dim=1)

        return out+x

class nn_pca_emulator:
    def __init__(self, 
                  model,
                  dv_fid, dv_std, cov_inv,
                  evecs,
                  device,
                  optim=None, reduce_lr=True, scheduler=None,
                  dtype='float'):

        if dtype=='double':
            torch.set_default_dtype(torch.double)
            print('default data type = double')
         
        self.optim     = optim
        self.device    = device 
        self.dv_fid    = torch.Tensor(dv_fid)
        self.cov_inv   = torch.Tensor(cov_inv)
        self.dv_std    = torch.Tensor(dv_std)
        self.evecs     = evecs
        self.reduce_lr = reduce_lr
        self.model     = model
        self.trained   = False
        
        if self.optim is None:
            self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-3 ,weight_decay=1e-3)
        if self.reduce_lr == True:
            print('Reduce LR on plateu: ',self.reduce_lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min',patience=10)#, factor=0.5)

    def train(self, X, y, X_validation, y_validation, test_split=None, batch_size=1000, n_epochs=150):
        summary(self.model)
        print('Batch size = ',batch_size)
        print('N_epochs = ',n_epochs)
        print('Begin training...')

        # ge normalization factors
        if not self.trained:
            self.X_mean = torch.Tensor(X.mean(axis=0, keepdims=True))
            self.X_std  = torch.Tensor(X.std(axis=0, keepdims=True))
            self.y_mean = self.dv_fid
            self.y_std  = self.dv_std

        # initialize arrays
        losses_train = []
        losses_vali = []
        loss = 100.

        # send everything to device
        self.model.to(self.device)
        tmp_y_std        = self.y_std.to(self.device)
        tmp_cov_inv      = self.cov_inv.to(self.device)
        tmp_X_mean       = self.X_mean.to(self.device)
        tmp_X_std        = self.X_std.to(self.device)
        tmp_X_validation = (X_validation.to(self.device) - tmp_X_mean)/tmp_X_std
        tmp_Y_validation = y_validation.to(self.device)

        # Here is the input normalization
        X_train     = ((X - self.X_mean)/self.X_std) # Note that this can mean inputs are <0 !!! Should not use non-symmetric activations
        y_train     = y
        trainset    = torch.utils.data.TensorDataset(X_train, y_train)
        validset    = torch.utils.data.TensorDataset(tmp_X_validation,tmp_Y_validation)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
        validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    
        train_start_time = datetime.now()
        for e in range(n_epochs):
            start_time = datetime.now()
            self.model.train()
            losses = []
            for i, data in enumerate(trainloader):    
                X       = data[0].to(self.device)
                Y_batch = data[1].to(self.device)
                Y_pred  = self.model(X) * tmp_y_std

                # PCA part
                diff = Y_batch - Y_pred
                loss1 = (diff \
                        @ tmp_cov_inv) \
                        @ torch.t(diff)

                loss = torch.mean(torch.diag(loss1))#+loss2))
                losses.append(loss.cpu().detach().numpy())
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            losses_train.append(np.mean(losses))
            ###validation loss
            with torch.no_grad():
                self.model.eval()
                losses = []
                for i, data in enumerate(validloader):  
                    X_v       = data[0].to(self.device)
                    Y_v_batch = data[1].to(self.device)

                    Y_v_pred = self.model(X_v) * tmp_y_std

                    v_diff = Y_v_batch - Y_v_pred 
                    loss_vali1 = (v_diff \
                                    @ tmp_cov_inv) @ \
                                    torch.t(v_diff)
                    loss_vali = torch.mean(torch.diag(loss_vali1))
     
                    losses.append(np.float(loss_vali.cpu().detach().numpy()))

                losses_vali.append(np.mean(losses))
                if self.reduce_lr:
                    self.scheduler.step(losses_vali[e])

            end_time = datetime.now()
            print('epoch {}, loss={}, validation loss={}, lr={} (epoch time: {})'.format(
                        e,
                        losses_train[-1],
                        losses_vali[-1],
                        self.optim.param_groups[0]['lr'],
                        (end_time-start_time).total_seconds()
                    ))#, total runtime: {} ({} average))
        
        np.savetxt("losses.txt", np.array([losses_train,losses_vali],dtype=np.float64))
        self.trained = True

    def predict(self, X):
        assert self.trained, "The emulator needs to be trained first before predicting"

        with torch.no_grad():
            y_pred = (self.model((X - self.X_mean) / self.X_std).double() * self.dv_std.double()).numpy() #normalization

        y_pred = y_pred @ np.linalg.inv(self.evecs)+ np.array([self.dv_fid.numpy()]) # convert back to data basis
        return y_pred

    def save(self, filename):
        torch.save(self.model, filename)
        with h5.File(filename + '.h5', 'w') as f:
            f['X_mean'] = self.X_mean
            f['X_std']  = self.X_std
            f['dv_fid'] = self.dv_fid
            f['dv_std'] = self.dv_std
            f['evecs']  = self.evecs
        
    def load(self, filename, device='cpu'):
        self.trained = True
        self.model = torch.load(filename,map_location=device)
        summary(self.model)
        self.model.eval()
        with h5.File(filename + '.h5', 'r') as f:
            self.X_mean = torch.Tensor(f['X_mean'][:]).double()
            self.X_std  = torch.Tensor(f['X_std'][:]).double()
            self.dv_fid = torch.Tensor(f['dv_fid'][:]).double()
            self.dv_std = torch.Tensor(f['dv_std'][:]).double()
            self.evecs  = torch.Tensor(f['evecs'][:]).double()


