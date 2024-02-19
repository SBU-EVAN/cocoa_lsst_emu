##NOTE: this check doesn't include fast parameters. Do check full prediction of emulator, please use lsst_emu_cs_lcdm.py in cobaya.likelihood


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
import cocoa_emu
from cocoa_emu import cocoa_config, nn_pca_emulator
from cocoa_emu.nn_emulator import ResBlock, Better_Attention, Better_Transformer, Affine
from torchinfo import summary
from datetime import datetime
import sys

def get_chi2(dv_predict, dv_exact, mask, cov_inv):
    delta_dv = (dv_predict - np.float32(dv_exact))[mask]
    chi2 = np.matmul( np.matmul(np.transpose(delta_dv), np.float32(cov_inv_masked)) , delta_dv  )   
    return chi2

# adjust config
configfile = './projects/lsst_y1/train_emulator.yaml'
config = cocoa_config(configfile)

# validation samples
# # T=256
# suffix = 1
# file = './projects/lsst_y1/emulator_output/chains/train_t256'

# # T=128
suffix = 0
file = './projects/lsst_y1/emulator_output/chains/valid_post_T64_atplanck'

# T=64
# suffix = 0
# file = './projects/lsst_y1/emulator_output/chains/valid_post_T64_atplanck' 


samples_validation = np.load(file+'_samples_'+str(suffix)+'.npy')
dv_validation      = np.load(file+'_data_vectors_'+str(suffix)+'.npy')[:,:780]#[::1,:780]

# thin
target_n = 10000
thin_factor = len(samples_validation)//target_n
if thin_factor!=0:
    samples_validation = samples_validation[::thin_factor]
    dv_validation      = dv_validation[::thin_factor]

# adjust as needed
OUTPUT_DIM=780

mask=config.mask[:OUTPUT_DIM]
cov_inv_masked = config.cov_inv_masked

logA   = samples_validation[:,0]
ns     = samples_validation[:,1]
H0     = samples_validation[:,2]
Omegab = samples_validation[:,3]
Omegac = samples_validation[:,4]

bin_count = 0
start_idx = 0
end_idx   = 0

# set needed parameters to initialize emulator
device=torch.device('cpu')
torch.set_num_threads(1) # `Intra-op parallelism
evecs=0

# get list of trained emus
#model_list = os.listdir('./projects/lsst_y1/emulator_output/models/new_trf/') #os.listdir('./projects/lsst_y1/emulator_output/models/for_tables/')
model_list = [
    'test'
]

in_dim=12
# N_layers = 1
int_dim_res = 256
n_channels = 32
int_dim_trf = 1024
out_dim = 780

layers = []
layers.append(nn.Linear(in_dim, int_dim_res))
layers.append(ResBlock(int_dim_res, int_dim_res))
layers.append(ResBlock(int_dim_res, int_dim_res))
layers.append(ResBlock(int_dim_res, int_dim_res))
# layers.append(ResBlock(int_dim_res, int_dim_res))
layers.append(nn.Linear(int_dim_res, int_dim_trf))
layers.append(Better_Attention(int_dim_trf, n_channels))
layers.append(Better_Transformer(int_dim_trf, n_channels))
layers.append(Better_Attention(int_dim_trf, n_channels))
layers.append(Better_Transformer(int_dim_trf, n_channels))
layers.append(Better_Attention(int_dim_trf, n_channels))
layers.append(Better_Transformer(int_dim_trf, n_channels))
layers.append(nn.Linear(int_dim_trf, out_dim))
layers.append(Affine())

nn_model = nn.Sequential(*layers)

results = []

for model in model_list:
    # open the trained emulator
    if '.h5' in model:
        continue # these files are not emulators

    print(model)
    emu_cs = nn_pca_emulator(nn_model, config.dv_fid, 0, cov_inv_masked, evecs, device=device, reduce_lr=True,lr=1e-3,weight_decay=1e-3)
    emu_cs.load('./projects/lsst_y1/emulator_output/models/'+model,state_dict=True)
    print('emulator(s) loaded\n')

    chi2_list=np.zeros(len(samples_validation))
    count_1 = 0 # for chi2>1
    count_2 = 0 # for chi2>0.2
    start_time=datetime.now()
    time_prev=start_time
    predicted_dv = np.zeros(OUTPUT_DIM)

    for j,point in enumerate(samples_validation):
        _j=j+1

        # get params and true dv
        theta = torch.Tensor(point)
        dv_truth = dv_validation[j]

        # reconstruct dv
        dv_cs = emu_cs.predict(theta[:12])[0]
        predicted_dv = dv_cs

        # compute chi2
        chi2 = get_chi2(predicted_dv, dv_truth, mask, cov_inv_masked)

        #count how many points have "poor" prediction.
        chi2_list[j] = chi2
        if chi2>1:
           count_1 += 1
        if chi2>0.2:
           count_2 += 1

        # progress check
        if j%10==0:
            runtime=datetime.now()-start_time
            print('\rprogress: '+str(j)+'/'+str(len(samples_validation))+\
                ' | runtime: '+str(runtime)+\
                ' | remaining time: '+str(runtime*(len(samples_validation)/_j - 1))+\
                ' | s/it: '+str(runtime/_j),end='')

    #summary
    #print("\naverage chi2 is: ", np.average(chi2_list))
    #print("Warning: This can be different from the training-validation loss. It depends on the mask file you use.")
    #print("points with chi2 > 0.25: "+str(count)+" ( "+str((count*100)/len(samples_validation))+"% )")

    print('\n model: ',model)
    print("average chi2 is: {:.3f}".format(np.mean(chi2_list)))
    print("median chi2 is: {:.3f} ".format(np.median(chi2_list)))
    print('num points: {}'.format(len(chi2_list)))
    print('numer of points chi2>1 {}'.format(count_1))
    print('numer of points chi2>0.2 {}\n'.format(count_2))
    
    np.savetxt('delta_chi2/'+model+'.txt',chi2_list)