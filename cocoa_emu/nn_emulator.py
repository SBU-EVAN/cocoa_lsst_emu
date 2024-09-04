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
from numba import njit

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gain + self.bias

class Better_ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(Better_ResBlock, self).__init__()
        
        if in_size != out_size: 
            self.skip = nn.Linear(in_size, out_size, bias=False) # we don't consider this. remove?
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()#torch.nn.BatchNorm1d(in_size)
        #self.norm2 = Affine()#torch.nn.BatchNorm1d(in_size)
        self.norm3 = Affine()#torch.nn.BatchNorm1d(in_size)

        self.act1 = activation_fcn(in_size) #nn.Tanh()#nn.ReLU()#
        #self.act2 = #nn.Tanh()#nn.ReLU()#
        self.act3 = activation_fcn(in_size) #nn.Tanh()#nn.ReLU()#

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.act1(self.norm1(self.layer1(x)))
        o2 = self.layer2(o1) + xskip             #(self.norm2(self.layer2(o1))) + xskip
        o3 = self.act3(self.norm3(o2))

        return o3

class ResBottle(nn.Module):
    def __init__(self, size, N):
        super(ResBottle, self).__init__()

        self.size = size
        self.N = N
        encoded_size = size // N

        # first layer
        self.norm1  = torch.nn.BatchNorm1d(encoded_size)
        self.layer1 = nn.Linear(size,encoded_size)
        self.act1   = nn.Tanh()

        # middle layer
        self.norm2  = torch.nn.BatchNorm1d(encoded_size)
        self.layer2 = nn.Linear(encoded_size,encoded_size)
        self.act2   = nn.Tanh()

        # last layer
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

class Better_Attention(nn.Module):
    def __init__(self, in_size ,n_partitions, dropout=False):
        super(Better_Attention, self).__init__()

        self.embed_dim    = in_size//n_partitions
        self.WQ           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WK           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WV           = nn.Linear(self.embed_dim,self.embed_dim)

        self.act          = nn.Softmax(dim=1) #NOT along the batch direction, apply to each vector.
        self.scale        = np.sqrt(self.embed_dim)
        self.n_partitions = n_partitions # n_partions or n_channels are synonyms 
        self.norm         = torch.nn.LayerNorm(in_size) # layer norm has geometric order (https://lessw.medium.com/what-layernorm-really-does-for-attention-in-transformers-4901ea6d890e)

        self.dropout = dropout
        if self.dropout:
            self.drop = nn.Dropout(p=0.1)
        else:
            self.drop = nn.Identity()

    def forward(self, x):
        x_norm    = self.norm(x)
        batch_size = x.shape[0]
        _x = x_norm.reshape(batch_size,self.n_partitions,self.embed_dim) # put into channels

        Q = self.WQ(_x) # query with q_i as rows
        K = self.WK(_x) # key   with k_i as rows
        V = self.WV(_x) # value with v_i as rows

        dot_product = torch.bmm(Q,K.transpose(1, 2).contiguous())
        normed_mat  = self.act(dot_product/self.scale)
        prod        = torch.bmm(normed_mat,V)

        #out = torch.cat(tuple([prod[:,i] for i in range(self.n_partitions)]),dim=1)+x
        out = self.drop(torch.reshape(prod,(batch_size,-1)))+x # reshape back to vector

        return out

class Better_Transformer(nn.Module):
    def __init__(self, in_size, n_partitions, dropout=False):
        super(Better_Transformer, self).__init__()  
    
        # get/set up hyperparams
        self.in_size      = in_size
        self.int_dim      = in_size//n_partitions 
        self.n_partitions = n_partitions
        self.act          = activation_fcn(in_size)  #nn.Tanh()   #nn.ReLU()#
        self.norm         = torch.nn.BatchNorm1d(in_size)
        #self.act2         = nn.Tanh()#nn.ReLU()#
        #self.norm2        = torch.nn.BatchNorm1d(in_size)
        self.act3         = activation_fcn(in_size)  #nn.Tanh()
        self.norm3        = torch.nn.BatchNorm1d(in_size)

        # set up weight matrices and bias vectors
        weights1 = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights1 = nn.Parameter(weights1) # turn the weights tensor into trainable weights
        bias1 = torch.Tensor(in_size)
        self.bias1 = nn.Parameter(bias1) # turn bias tensor into trainable weights

        weights2 = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights2 = nn.Parameter(weights2) # turn the weights tensor into trainable weights
        bias2 = torch.Tensor(in_size)
        self.bias2 = nn.Parameter(bias2) # turn bias tensor into trainable weights

        # initialize weights and biases
        # this process follows the standard from the nn.Linear module (https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77)
        nn.init.kaiming_uniform_(self.weights1, a=np.sqrt(5)) # matrix weights init 
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weights1) # fan_in in the input size, fan out is the output size but it is not use here
        bound1 = 1 / np.sqrt(fan_in1) 
        nn.init.uniform_(self.bias1, -bound1, bound1) # bias weights init

        nn.init.kaiming_uniform_(self.weights2, a=np.sqrt(5))  
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weights2)
        bound2 = 1 / np.sqrt(fan_in2) 
        nn.init.uniform_(self.bias2, -bound2, bound2)

        self.dropout = dropout
        if self.dropout:
            self.drop = nn.Dropout(p=0.1)
        else:
            self.drop = nn.Identity()

    def forward(self,x):
        mat1 = torch.block_diag(*self.weights1) # how can I do this on init rather than on each forward pass?
        mat2 = torch.block_diag(*self.weights2)
        #x_norm = self.norm(x)
        #_x = x_norm.reshape(x_norm.shape[0],self.n_partitions,self.int_dim) # reshape into channels
        #_x = x.reshape(x.shape[0],self.n_partitions,self.int_dim) # reshape into channels

        # o1 = self.act(self.norm(torch.matmul(x,mat1)+self.bias1))
        # o2 = torch.matmul(o1,mat2)+self.bias2  #self.act2(self.norm2(torch.matmul(o1,mat2)+self.bias2))
        # o3 = self.act3(self.norm3(o2+x))
        # return o3

        # TEST ACTIVATION FUNCTION #
        o1 = self.norm(torch.matmul(x,mat1)+self.bias1)
        o2 = self.act(o1)#.reshape(x.shape[0],self.n_partitions,self.int_dim)).reshape(x.shape[0],self.in_size)
        o3 = self.drop(torch.matmul(o1,mat2) + self.bias2) + x
        o4 = self.act3(o3)#.reshape(x.shape[0],self.n_partitions,self.int_dim)).reshape(x.shape[0],self.in_size)
        return o4
        # END ACTIVATION TEST

class activation_fcn(nn.Module):
    def __init__(self, dim):
        super(activation_fcn, self).__init__()

        self.dim = dim
        self.gamma = nn.Parameter(torch.zeros((dim)))
        self.beta = nn.Parameter(torch.zeros((dim)))

    def forward(self,x):
        #exp = -1*torch.mul(self.beta,x)
        #inv = (1+torch.exp(exp)).pow_(-1)
        exp = torch.mul(self.beta,x)
        inv = torch.special.expit(exp)
        fac_2 = 1-self.gamma
        out = torch.mul(self.gamma + torch.mul(inv,fac_2), x)
        return out

class True_Transformer(nn.Module):
    def __init__(self, in_size, n_partitions):
        super(True_Transformer, self).__init__()  
    
        self.int_dim      = in_size//n_partitions
        self.n_partitions = n_partitions
        self.linear       = nn.Linear(self.int_dim,self.int_dim)#ResBlock(self.int_dim,self.int_dim)#
        self.act          = nn.ReLU()
        self.norm         = torch.nn.BatchNorm1d(self.int_dim*n_partitions)

    def forward(self,x):
        batchsize = x.shape[0]
        out = torch.reshape(self.norm(x),(batchsize,self.n_partitions,self.int_dim))
        out = self.act(self.linear(out))
        out = torch.reshape(out,(batchsize,self.n_partitions*self.int_dim))
        return out+x



class nn_pca_emulator:
    def __init__(self, 
                  model,
                  dv_fid, dv_std, cov_inv,
                  evecs,
                  device,
                  optim=None, lr=1e-3, reduce_lr=True, scheduler=None,
                  weight_decay=1e-3,
                  dtype='float'):
         
        self.optim        = optim
        self.device       = device 
        self.dv_fid       = torch.Tensor(dv_fid)
        self.cov_inv      = torch.Tensor(cov_inv)
        self.dv_std       = torch.Tensor(dv_std)
        self.evecs        = evecs
        self.reduce_lr    = reduce_lr
        self.model        = model
        self.trained      = False
        self.weight_decay = weight_decay
        
        if self.optim is None:
            print('Learning rate = {}'.format(lr))
            print('Weight decay = {}'.format(weight_decay))
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr ,weight_decay=self.weight_decay)
            #self.optim = torch.optim.SGD(self.model.parameters(), lr=lr ,weight_decay=self.weight_decay)
        if self.reduce_lr == True:
            print('Reduce LR on plateu: ',self.reduce_lr)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min',patience=15,factor=0.1)#,min_lr=1e-12)#, factor=0.5)
            #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, 0.95, last_epoch=-1)
        if dtype=='double':
            torch.set_default_dtype(torch.double)
            print('default data type = double')
        if device!=torch.device('cpu'):
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.generator=torch.Generator(device=self.device)

    def train(self, X, y, X_validation, y_validation, test_split=None, batch_size=1000, n_epochs=150):
        summary(self.model)
        print('Batch size = ',batch_size)
        print('N_epochs = ',n_epochs)

        # get normalization factors
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
        tmp_Y_validation = y_validation.to(self.device)#(y_validation.to(self.device) - tmp_X_mean)/tmp_X_std#y_validation.to(self.device)

        # Here is the input normalization
        X_train     = ((X - self.X_mean)/self.X_std)
        y_train     = y#((y - self.X_mean)/self.X_std)#y
        trainset    = torch.utils.data.TensorDataset(X_train, y_train)
        validset    = torch.utils.data.TensorDataset(tmp_X_validation,tmp_Y_validation)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0, generator=self.generator)
        validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0, generator=self.generator)
    
        print('Datasets loaded!')
        print('Begin training...')
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

                ### remove the largest from each batch (ES TESTING!)
                # loss_arr = torch.diag(loss1)
                # sort_loss = torch.sort(loss_arr)
                # loss = torch.mean(sort_loss[0][:-5])
                ### END TESTING
                ### BEGIN TESTING
                #loss = torch.mean(torch.log(torch.diag(loss1))) # log chi^2
                #loss = torch.mean(torch.log(1+torch.diag(loss1))) # log hyperbola
                loss = torch.mean((1+2*torch.diag(loss1))**(1/2))-1 #hyperbola
                #loss = torch.mean((1+3*torch.diag(loss1))**(1/3))-1 #hyperbola^1/3
                ### END TESTING

                #loss = torch.mean(torch.diag(loss1)) # commented out for testing
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

                    ### remove the largest 2 from each batch (ES TESTING!)
                    # loss_vali_arr = torch.diag(loss_vali1)
                    # sort_vali_loss = torch.sort(loss_vali_arr)
                    # loss_vali = torch.mean(sort_vali_loss[0][:-5])
                    ### END TESTING
                    ### BEGIN TESTING
                    #loss_vali = torch.mean(torch.log(torch.diag(loss_vali1))) # log chi^2
                    #loss_vali = torch.mean(torch.log(1+torch.diag(loss_vali1))) # log hyperbola
                    loss_vali = torch.mean((1+2*torch.diag(loss_vali1))**(1/2))-1 #hyperbola
                    #loss_vali = torch.mean((1+3*torch.diag(loss_vali1))**(1/3))-1 #hyperbola^1/3
                    ### END TESTING
                    
                    #loss_vali = torch.mean(torch.diag(loss_vali1)) # commented out and replaced with testing portion above
                    losses.append(np.float(loss_vali.cpu().detach().numpy()))

                losses_vali.append(np.mean(losses))
                if self.reduce_lr:
                    self.scheduler.step(losses_vali[e])

                self.optim.zero_grad()

            end_time = datetime.now()
            print('epoch {}, loss={:.5f}, validation loss={:.5f}, lr={:.2E} (epoch time: {:.1f})'.format(
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
            y_pred = (self.model((X - self.X_mean) / self.X_std) * self.dv_std) #normalization

        y_pred = y_pred @ torch.linalg.inv(self.evecs)+ self.dv_fid # convert back to data basis
        return y_pred.cpu().detach().numpy()

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        with h5.File(filename + '.h5', 'w') as f:
            f['X_mean'] = self.X_mean
            f['X_std']  = self.X_std
            f['dv_fid'] = self.dv_fid
            f['dv_std'] = self.dv_std
            f['evecs']  = self.evecs
        
    def load(self, filename, device=torch.device('cpu'),state_dict=False):
        self.trained = True
        if device!=torch.device('cpu'):
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        if state_dict==False:
            self.model = torch.load(filename,map_location=device)
        else:
            print('Loading with "torch.load_state_dict(torch.load(file))"...')
            self.model.load_state_dict(torch.load(filename,map_location=device))
        #summary(self.model)
        self.model.eval()
        with h5.File(filename + '.h5', 'r') as f:
            self.X_mean = torch.Tensor(f['X_mean'][:])
            self.X_std  = torch.Tensor(f['X_std'][:])
            self.dv_fid = torch.Tensor(f['dv_fid'][:])
            self.dv_std = torch.Tensor(f['dv_std'][:])
            self.evecs  = torch.Tensor(f['evecs'][:])



################ OLD FCNS, KEPT FOR COMPLETENESS

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
        self.norm         = torch.nn.LayerNorm(in_size)#BatchNorm1d(in_size)

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

class ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()
        
        if in_size != out_size: 
            self.skip = nn.Linear(in_size, out_size, bias=False) # we don't consider this. remove?
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()
        self.norm2 = Affine()

        self.act1 = nn.Tanh()#nn.ReLU()#
        self.act2 = nn.Tanh()#nn.ReLU()#

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.layer1(self.act1(self.norm1(x))) / np.sqrt(10)
        o2 = self.layer2(self.act2(self.norm2(o1))) / np.sqrt(10) + xskip

        return o2


#####   STUFF FOR MAMBA   #####
class SSM(nn.Module):
    def __init__(self,
            input_dim, device
        ):
        super(SSM, self).__init__()
        self.input_dim = input_dim
        self.device = device
        # self.A = nn.Linear(input_dim,1)
        # self.B = nn.Linear(1,input_dim,bias=False)
        # self.C = nn.Linear(input_dim,1,bias=False)
        # self.D = nn.Linear(1,1,bias=False)

        # # construct the 'convolution' kernel???
        # # to do this i need to intialize weights for A, B, C, D (1, 2, 3, 4)
        # weights1 = torch.zeros((input_dim))
        # self.weights1 = nn.Parameter(weights1) # turn the weights tensor into trainable weights
        # #bias1 = torch.Tensor(in_size)
        # #self.bias1 = nn.Parameter(bias1) # turn bias tensor into trainable weights

        # weights2 = torch.zeros((input_dim,1))
        # self.weights2 = nn.Parameter(weights2) # turn the weights tensor into trainable weights
        # #bias2 = torch.Tensor(in_size)
        # #self.bias2 = nn.Parameter(bias2) # turn bias tensor into trainable weights

        # weights3 = torch.zeros((1,input_dim))
        # self.weights3 = nn.Parameter(weights3) # turn the weights tensor into trainable weights
        # #bias3 = torch.Tensor(in_size)
        # #self.bias3 = nn.Parameter(bias3) # turn bias tensor into trainable weights

        # weights4 = torch.zeros((1))
        # self.weights4 = nn.Parameter(weights4) # turn the weights tensor into trainable weights
        # #bias4 = torch.Tensor(in_size)
        # #self.bias4 = nn.Parameter(bias4) # turn bias tensor into trainable weights

        # # initialize weights and biases
        # # this process follows the standard from the nn.Linear module (https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77)
        # #nn.init.kaiming_uniform_(self.weights1, a=np.sqrt(5)) # matrix weights init 
        # #fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weights1) # fan_in in the input size, fan out is the output size but it is not use here
        # #bound1 = 1 / np.sqrt(fan_in1) 
        # #nn.init.uniform_(self.bias1, -bound1, bound1) # bias weights init

        # nn.init.kaiming_uniform_(self.weights2, a=np.sqrt(5))  
        # #fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weights2)
        # #bound2 = 1 / np.sqrt(fan_in2) 
        # #nn.init.uniform_(self.bias2, -bound2, bound2)

        # nn.init.kaiming_uniform_(self.weights3, a=np.sqrt(5))  
        # #fan_in3, _ = nn.init._calculate_fan_in_and_fan_out(self.weights3)
        # #bound3 = 1 / np.sqrt(fan_in3) 
        # #nn.init.uniform_(self.bias3, -bound3, bound3)

        # #nn.init.kaiming_uniform_(self.weights4, a=np.sqrt(5))  
        # #fan_in4, _ = nn.init._calculate_fan_in_and_fan_out(self.weights4)
        # #bound4 = 1 / np.sqrt(fan_in4) 
        # #nn.init.uniform_(self.bias4, -bound4, bound4)

        ### using convolution kernel
        # self.K = torch.zeros((self.input_dim))
        # #logA = torch.log(weights1)
        # if self.switch==0:
        #     for i in range(self.input_dim):
        #         pow_A = torch.diag(self.weights1**i)
        #         elem = self.weights3 @ pow_A @ self.weights2
        #         self.K[i] += elem[0,0]


        weights1 = torch.zeros((input_dim))
        self.weights1 = nn.Parameter(weights1)
        nn.init.uniform_(self.weights1,-1/np.sqrt(input_dim),1/np.sqrt(input_dim))

        weights2 = torch.zeros((input_dim))
        self.weights2 = nn.Parameter(weights2)
        nn.init.uniform_(self.weights2,-1/np.sqrt(input_dim),1/np.sqrt(input_dim))

        weights3 = torch.zeros((input_dim))
        self.weights3 = nn.Parameter(weights3)
        nn.init.uniform_(self.weights3,-1/np.sqrt(input_dim),1/np.sqrt(input_dim))

        self.bias1 = nn.Parameter(torch.ones(1))
        self.bias2 = nn.Parameter(torch.ones(1))

        #self.pows = torch.arange(0,self.input_dim).reshape((1,self.input_dim)).to(device)
        #print(pows)
        #self.pows = self.pows.expand(self.input_dim,self.input_dim)#.T.reshape(self.input_dim**2)

        self.delta = nn.Linear(self.input_dim,1)
        self.act   = nn.Softplus()

    def forward(self, u):
        #starttime=datetime.now()
        batchsize = u.shape[0]
        # # get the latent space state vector 0
        # u0 = u[:,0] 
        # u0 = torch.reshape(u0,(batchsize,1,1))
        # xi = self.B(u0)
        # y = torch.zeros(u.shape) 
        # y[:,0] = xi[:,0,0]

        # # primitive implementation
        # for i in range(1,u.shape[1]):
        #     ui = torch.reshape(u[:,i],(batchsize,1,1))
        #     #xi = w1 @ xi + self.B(ui) 
        #     #yi = self.C(w1 @ xi + self.B(ui) ) + self.D(ui)
        #     y[:,i] += (self.C(self.weights1*xi + self.B(ui) ) + self.D(ui))[0,0]


        # implement with the selection mechanism
        # first the selection mechanism
        select = self.act(self.delta(u) + self.bias1)
        # now compute A and B
        A = self.weights1 * u
        Bbar = self.weights2 * u 
        # compute Abar from A, delta, and activation
        Abar = select * A
        Cbar = self.weights3 * u 
        Dbar = self.bias2[0]
        #print('abar',Abar.shape)
        #print('Bbar',Bbar.shape)

        y = self.compute_recursive(u, Abar, Bbar, Cbar, Dbar)



        ### using convolution kernel

        # setup the kernel
        # first the selection mechanism
        # select = self.act(self.delta(u) + self.bias1)
        # # now compute A and B
        # A = self.weights1 * u
        # B = self.weights2 * u 
        # # compute Abar from A, delta, and activation
        # Abar = select * A
        # Abar = Abar.expand(self.input_dim,self.input_dim)
        # Abar = torch.exp(Abar)**self.pows

        # # construct the kernel from C, Abar^k, Bbar
        # K = torch.sum(self.weights3 * Abar.T * self.weights2,axis=1).reshape((1,1,self.input_dim))

        # # now lets do the convolution
        # u_ = nn.functional.pad(input=u,pad=(self.input_dim-1,0,0,0),mode='constant',value=0)
        # u_ = torch.reshape(u_,(batchsize,1,2*self.input_dim-1))
        # y = torch.reshape(torch.nn.functional.conv1d(u_,K),(batchsize,self.input_dim))
        # y += self.bias2*u

        return y

    @torch.jit.script
    def compute_recursive(u, A, B, C, D):
        batchsize = u.shape[0]
        u0 = u[:,0] 
        xi = torch.einsum('bi,b->b',B,u0)
        #print('x0',xi.shape) # (batchsize x 1) matrix
        
        y = torch.zeros(u.shape).to('cuda') 
        y[:,0] = xi

        #print('A',A.shape)
        #print('B',B.shape)
        #print('C',C.shape)
        #print('D',D.shape)

        #print('u',u.shape)
        #print('x0',xi.shape)

        #xi = torch.reshape(xi,(batchsize,1))

        # primitive implementation
        for i in range(1,u.shape[1]):
            #ui = torch.reshape(u[:,i],(batchsize,1,1))
            #xi = w1 @ xi + self.B(ui) 
            #yi = self.C(w1 @ xi + self.B(ui) ) + self.D(ui)

            #    y[:,i] += (self.C(self.weights1*xi + self.B(ui) ) + self.D(ui))[0,0]

            ui = u[:,i]#torch.reshape(u[:,i],(batchsize,1))
            #print('ui',ui.shape)
            xi = torch.einsum('bi,b->b',A,xi) + torch.einsum('bi,b->b',B,ui)
            #print('xi',xi.shape)
            y[:,i] += torch.einsum('bi,b->b',C,xi) + D * ui

        return y



class mamba_block(nn.Module):
    def __init__(self, 
            input_dim, int_dim, 
            in_channels, out_channels, kernel_size, stride, 
            n_partitions, device):
        super(mamba_block, self).__init__()

        self.input_dim    = input_dim
        self.int_dim      = int_dim
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.n_partitions = n_partitions

        self.linear1 = nn.Linear(input_dim, int_dim)
        self.linear2 = nn.Linear(input_dim, int_dim)
        self.linear3 = nn.Linear(int_dim,input_dim)

        self.act1 = nn.SiLU() #nn.Tanh()
        self.act2 = nn.SiLU() #nn.Tanh()

        self.act3 = nn.SiLU() # is this implicit just like every other ML paper? >:(

        # mamba puts a norm and residual between each block. I will do that here to simplify the code.
        self.norm = nn.LayerNorm(input_dim)

        # we need a padding so that the convolution output size is equivalent to the input size
        self.padding = (kernel_size-1)//2
        assert self.padding == (kernel_size-1)/2


        # make a few assertions to ensure dimensions will work out
        # self.conv_out_dim = 1 + (int_dim - kernel_size)/stride
        # self.attention_dim = self.conv_out_dim/n_partitions
        # print(self.conv_out_dim)
        # assert self.conv_out_dim - int(self.conv_out_dim) == 0 # no data is thrown away
        # assert self.conv_out_dim > self.input_dim # no compression of the input
        # assert self.attention_dim - int(self.attention_dim) == 0 #no data thrown out in attn

        self.conv = nn.Conv1d(self.in_channels,self.out_channels,kernel_size=kernel_size,stride=stride,padding=self.padding)
        self.ssm = SSM(self.int_dim,device) 
        # self.ssm = Better_Attention(self.int_dim,self.n_partitions) # There are others that are much more sophisticated, use this for testing for now

    def forward(self,x):
        # 'left' side of mamba
        batchsize = x.shape[0]
        x1 = self.linear1(x)
        x1 = torch.reshape(x1,(batchsize,1,self.int_dim))
        x1 = self.conv(x1)
        x1 = torch.reshape(x1,(batchsize,self.int_dim))
        x1 = self.act1(x1)
        x1 = self.ssm(x1)

        # 'right' side of mamba
        x2 = self.linear2(x)
        x2 = self.act2(x2)

        #multiply elementwise
        y = torch.mul(x1,x2)

        #project to output_dim=input_dim
        y = self.linear3(y)

        return self.act3(self.norm(y)+x)

class Old_Better_Transformer(nn.Module):
    def __init__(self, in_size, n_partitions):
        super(Old_Better_Transformer, self).__init__()  

        # get/set up hyperparams
        self.int_dim      = in_size//n_partitions 
        self.n_partitions = n_partitions
        self.act          = activation_fcn(self.int_dim)#nn.Tanh()#nn.ReLU()#
        self.norm         = torch.nn.BatchNorm1d(in_size)

        # set up weight matrices and bias vectors
        weights = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights = nn.Parameter(weights) # turn the weights tensor into trainable weights
        bias = torch.Tensor(in_size)
        self.bias = nn.Parameter(bias) # turn bias tensor into trainable weights

        # initialize weights and biases
        # this process follows the standard from the nn.Linear module (https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77)
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5)) # matrix weights init 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights) # fan_in in the input size, fan out is the output size but it is not use here
        bound = 1 / np.sqrt(fan_in) 
        nn.init.uniform_(self.bias, -bound, bound) # bias weights init

    def forward(self,x):
        mat = torch.block_diag(*self.weights) # how can I do this on init rather than on each forward pass?
        x_norm = self.norm(x)
        _x = x_norm.reshape(x_norm.shape[0],self.n_partitions,self.int_dim) # reshape into channels
        o = self.act(torch.matmul(x_norm,mat)+self.bias)
        return o+x




