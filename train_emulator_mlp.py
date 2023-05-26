import sys,os
import numpy as np
import torch
import torch.nn as nn


sys.path.insert(0, os.path.abspath(".."))

from cocoa_emu import Config
from cocoa_emu import nn_pca_emulator 
from cocoa_emu.nn_emulator import Affine, ResBlock, ResBottle, DenseBlock, Attention, Transformer

if '--auto' in sys.argv:
    idx = sys.argv.index('--auto')
    print('running in automatic mode')

    INT_DIM = int(sys.argv[idx+1])
    dim_frac = int(sys.argv[idx+2])
    N_layers = int(sys.argv[idx+3])

    print('internal dimension:', INT_DIM)
    print('bottlneck factor:  ', dim_frac)
    print('number of layers:  ', N_layers)

else:
    INT_DIM = 128
    N=0

################################
#                              #
#    DEFINE YOUR MODEL HERE    #
#                              #
################################

N_DIM=12
# N_layers = 1
# INT_DIM  = 256
# dim_frac = 8
OUTPUT_DIM = 780

layers = []
layers.append(nn.Linear(N_DIM,INT_DIM))
layers.append(nn.Tanh())
for i in range(N_layers):
    layers.append(nn.Linear(INT_DIM,INT_DIM))
    nn.Tanh()
layers.append(nn.Linear(INT_DIM,OUTPUT_DIM))
layers.append(Affine())

model = nn.Sequential(*layers)

#===============================#

# open yaml config. 
configfile = sys.argv[1]
config = Config(configfile)

# Training set filenames
train_samples_files = sys.argv[2]
file = sys.argv[2]

### PARSE THE COMMAND LINE ARGS ###

### GET DATAVECTORS
i=0
if "-f" in sys.argv:
    idx = sys.argv.index('-f')
    dv_root = './projects/lsst_y1/emulator_output/chains/'
    for file in os.listdir(dv_root):
        if sys.argv[idx+1] in file:
            if 'samples' in file:
                if i==0:
                    print('Opening:',file)
                    train_samples = np.load(dv_root+file)
                    file = file.replace('samples','data_vectors')
                    print('Opening:',file)
                    train_data_vectors = np.load(dv_root+file)
                    i=1
                else: 
                    print('Opening:',file)
                    train_samples = np.vstack((train_samples, np.load(dv_root+file)))
                    file = file.replace('samples','data_vectors')
                    print('Opening:',file)
                    train_data_vectors = np.vstack((train_data_vectors, np.load(dv_root+file)))

#output file
if "-o" in sys.argv:
    idx = sys.argv.index('-o')
    outpath = sys.argv[idx+1]

#### adjust validation root directories to your sample directory
# You can add more probes. These are used when cutting up your data vector.
# They do NOT consider cross correlations between the components. Thus the total delta chi^2 is not the sum of the loss of each model
if config.probe=='cosmic_shear':
    print("training for cosmic shear only")
    start=0
    stop=780
    sample_dim=12
    validation_root='./projects/lsst_y1/emulator_output/chains/train_post_T64_none'
elif config.probe=='3x2pt':
    # 3x2pt is generally very difficult.
    print("trianing for 3x2pt")
    start=0
    stop=1560
    validation_root='./projects/lsst_y1/emulator_output/chains/vali_post_T1_3x2'
elif config.probe=='2x2pt':
    print("training for 2x2")
    start=780
    stop=1560
    validation_root='./projects/lsst_y1/emulator_output/chains/vali_post_T1_3x2'
else:
    print('probe not defined')
    quit()

train_data_vectors = train_data_vectors[:,start:stop]
train_samples = train_samples[:,:sample_dim]
cov_inv = np.linalg.inv(config.cov)[start:stop, start:stop] #NO mask here for cov_inv enters training
mask_cs = config.mask[start:stop]
dv_fid =config.dv_fid[start:stop]
dv_std = config.dv_std[start:stop]

def get_chi_sq_cut(train_data_vectors, chi2_cut):
    ### Use to apply chi2 cuts to data vectors which enter training. Not strictly necessary but training on a high chi2 range is more difficult
    chi_sq_list = []
    for dv in train_data_vectors:
        if config.probe=='cosmic_shear':
            delta_dv = (dv - config.dv_obs[0:OUTPUT_DIM])[mask_cs] #technically this should be masked(on a fiducial scale cut), but the difference is small
            chi_sq = delta_dv @ cov_inv[mask_cs][:,mask_cs] @ delta_dv
        elif config.probe=='3x2pt':
            delta_dv = (dv - config.dv_obs)[config.mask]
            chi_sq = delta_dv @ config.masked_inv_cov @ delta_dv
        if config.probe=='2x2pt':
            delta_dv = (dv - config.dv_obs[start:stop])[mask_2x2]
            chi_sq = delta_dv @ cov_inv[mask_2x2][:,mask_2x2] @ delta_dv

        chi_sq_list.append(chi_sq)
    chi_sq_arr = np.array(chi_sq_list)
    select_chi_sq = (chi_sq_arr < chi2_cut)
    return select_chi_sq

###============= Setting up validation set ============
validation_samples      = np.load(validation_root+'_samples_0.npy')[::60,:12] # careful with thinning!
validation_data_vectors = np.load(validation_root+'_data_vectors_0.npy')[::60,start:stop] #Thin only to number of validation dvs you want!

##### shuffeling
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    samples = a[p]
    dvs     = b[p]
    return samples, dvs

train_samples, train_data_vectors = unison_shuffled_copies(train_samples, train_data_vectors)
validation_samples, validation_data_vectors = unison_shuffled_copies(validation_samples, validation_data_vectors)

# Convert to eigenbasis if PCA is true in config.
# This greatly simplifies the information the NN needs to learn.
if config.do_PCA:
    lsst_cov = config.cov[start:stop,start:stop]
    dv_fid = config.dv_fid[start:stop]

    # do diagonalization
    eigensys = np.linalg.eigh(lsst_cov)
    evals = eigensys[0]
    evecs = eigensys[1]

    # Now we change basis to the eigenbasis + do the normalization
    dv = np.array([dv_fid for _ in range(len(train_data_vectors))])
    train_data_vectors = np.transpose((np.linalg.inv(evecs) @ np.transpose(train_data_vectors - dv)))
    dv = np.array([dv_fid for _ in range(len(validation_data_vectors))])
    validation_data_vectors = np.transpose((np.linalg.inv(evecs) @ np.transpose(validation_data_vectors - dv)))

    # compute the diagonalized cov
    cov_inv_pc = np.diag(1/evals)#np.linalg.inv(lsst_cov)
    dv_std = np.sqrt(evals)

print("Number of training points:  ", len(train_samples))
print("Number of validation points:", len(validation_samples))

# cuda or cpu
print('cuda device avaliable?  ',torch.cuda.is_available())
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    torch.set_num_interop_threads(40) # Inter-op parallelism
    torch.set_num_threads(40) # Intra-op parallelism

print('Using device: ',device)
    
TS = torch.Tensor(train_samples)
TDV = torch.Tensor(train_data_vectors)
VS = torch.Tensor(validation_samples)
VDV = torch.Tensor(validation_data_vectors)

emu = nn_pca_emulator(model,
                        dv_fid, dv_std, cov_inv_pc,
                        evecs, device, reduce_lr=True)

emu.train(TS, TDV, VS, VDV, batch_size=2500,n_epochs=100)# n_epochs=config.n_epochs)
print("model saved to ",str(config.savedir))
#emu.save(config.savedir + '/for_tables/'+str(config.probe)+'_nlayer_'+str(N_layers)+'_intdim_'+str(INT_DIM)+'_frac_'+str(dim_frac)) # Rename your model :)
emu.save(config.savedir+'/'+outpath)
print("DONE!!")   


##################################
#                                #
#      SOME GOOD MODELS :)       #
#                                #
##################################

#---     Resnet+Attention     ---#

# layers = []
# layers.append(ResBlock(INT_DIM,INT_DIM))
# layers.append(ResBlock(INT_DIM,INT_DIM))
# layers.append(ResBlock(INT_DIM,INT_DIM))
# layers.append(nn.Linear(INT_DIM,1024))
# layers.append(Attention(1024,dim_frac))
# layers.append(Transformer(dim_frac,1024//dim_frac))
# layers.append(nn.Linear(1024,OUTPUT_DIM))
# layers.append(Affine())

#--------------------------------#

##################################



