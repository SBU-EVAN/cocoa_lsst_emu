##NOTE: this check doesn't include fast parameters. Do check full prediction of emulator, please use lsst_emu_cs_lcdm.py in cobaya.likelihood


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from cocoa_emu import Config, nn_pca_emulator
from cocoa_emu.sampling import EmuSampler
from torchinfo import summary

# just for convinience
from datetime import datetime

#compute using double
#torch.set_default_dtype(torch.double)

def get_chi2(dv_predict, dv_exact, mask, cov_inv):
    delta_dv = (dv_predict - np.float32(dv_exact))[mask]
    chi2 = np.matmul( np.matmul(np.transpose(delta_dv), np.float32(cov_inv)) , delta_dv  )   
    return chi2

# adjust config
configfile = './projects/lsst_y1/train_emulator.yaml'
config = Config(configfile)

#which T you want to use
T_test = 8

# open validation samples
# !!! Watch thin factor !!!
samples_validation = np.load('./projects/lsst_y1/emulator_output/chains/train_post_T64_none_samples_2.npy')
dv_validation      = np.load('./projects/lsst_y1/emulator_output/chains/train_post_T64_none_data_vectors_2.npy')[:,:780]#[::1,:780]

# cut the data to the IA prior
#keep_idxs_1 = np.where(np.abs(samples_validation[:,10])<5)
#keep_idxs_2 = np.where(np.abs(samples_validation[:,11])<5)
#samples_validation = samples_validation[keep_idxs_1[0]]
#samples_validation = samples_validation[keep_idxs_2[0]]
#dv_validation      = dv_validation[keep_idxs_1[0]]
#dv_validation      = dv_validation[keep_idxs_2[0]]

# thin
target_n = 10000
thin_factor = len(samples_validation)//target_n
if thin_factor!=0:
    samples_validation = samples_validation[::thin_factor]
    dv_validation      = dv_validation[::thin_factor]

# output dim for full 3x2
# adjust as needed
OUTPUT_DIM=780
BIN_SIZE=OUTPUT_DIM
mask=config.mask[:OUTPUT_DIM]

cov            = config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM]
cov_inv        = np.linalg.inv(config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM])
cov_inv_masked = np.linalg.inv(config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM][mask][:,mask])

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
torch.set_num_threads(80) # `Intra-op parallelism
evecs=0

# get list of trained emus
#model_list = os.listdir('./projects/lsst_y1/emulator_output/models/temp_check/') #os.listdir('./projects/lsst_y1/emulator_output/models/for_tables/')
model_list = ['attention_transformer_test_500_epochs']
results = []

for model in model_list:
    #=====   open trained emulators   =====#
    if '.h5' in model:
        continue
    #if 'resnet' not in model or 'resbottle' in model:
    # if 'mlp' not in model:
    #     continue
    # if str(T_test) not in model:
    #     continue
    if 'resbottle' in model:
        continue
    print(model)
    emu_cs = nn_pca_emulator(nn.Sequential(nn.Linear(1,1)), config.dv_fid, config.dv_std, cov_inv, evecs, device)#,dtype='double')
    emu_cs.load('./projects/lsst_y1/emulator_output/models/'+model)#'./projects/lsst_y1/emulator_output/models/for_tables/'+model)#'projects/lsst_y1/emulator_output/models/model_T16')
    emu_cs.model = emu_cs.model.double()
    print('emulator(s) loaded\n')

    chi2_list=np.zeros(len(samples_validation))
    count=0
    start_time=datetime.now()
    time_prev=start_time
    predicted_dv = np.zeros(OUTPUT_DIM)

    for j in range(len(samples_validation)):
        _j=j+1

        # get params and true dv
        theta = torch.Tensor(samples_validation[j])
        dv_truth = dv_validation[j]

        # reconstruct dv
        dv_cs = emu_cs.predict(theta[:12])[0]
        predicted_dv = dv_cs

        # compute chi2
        chi2 = get_chi2(predicted_dv, dv_truth, mask, cov_inv_masked)

        #count how many points have "poor" prediction.
        chi2_list[j] = chi2
        if chi2>0.25:
           count += 1

        # progress check
        if j%10==0:
            runtime=datetime.now()-start_time
            print('\rprogress: '+str(j)+'/'+str(len(samples_validation))+\
                ' | runtime: '+str(runtime)+\
                ' | remaining time: '+str(runtime*(len(samples_validation)/_j - 1))+\
                ' | s/it: '+str(runtime/_j),end='')

    #summary
    print("\naverage chi2 is: ", np.average(chi2_list))
    print("Warning: This can be different from the training-validation loss. It depends on the mask file you use.")
    print("points with chi2 > 0.25: "+str(count)+" ( "+str((count*100)/len(samples_validation))+"% )")

    tmp_res = [model, np.average(chi2_list),(count*100)/len(samples_validation)]
    results.append(tmp_res)
    np.save('delta_chi2_data/'+model+'_deltachi2.npy',chi2_list)

    ###PLOT chi2 start
    cmap = plt.cm.get_cmap('coolwarm')

    num_bins = 100
    plt.xlabel(r'$\chi^2$')
    plt.ylabel('distribution')
    plt.xscale('log')

    plt.hist(chi2_list, num_bins, 
                                density = 1, 
                                color ='green',
                                alpha = 0.7)


    plt.savefig("T128_chi2.pdf")

    ####PLOT chi2 end

    #####PLOT 2d start######
    plt.figure().clear()
    file = open('model_table_results.txt','w')
    for res in results:
        file.write(str(res[0])+","+str(res[1])+","+str(res[2])+"\n")
    file.close()

    #plt.scatter(logA, Omegam, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap)
    plt.scatter(logA, Omegac, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa (T=64 chain)', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
    #plt.scatter(Omegam, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
    #plt.scatter(Omegam, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap)
    #plt.scatter(logA, Omegam_growth, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())
    #plt.scatter(H0, Omegab, c=chi2_list, label=r'$\chi^2$ between emulator and cocoa', s = 2, cmap=cmap,norm=matplotlib.colors.LogNorm())

    cb = plt.colorbar()

    plt.xlabel(r'$\log A$')
    plt.ylabel(r'$\Omega_c h^2$')

    plt.legend()
    #plt.savefig("T64_crazy.pdf")

    np.savetxt('./delta_chi2_T64/'+str(model)+'.txt',chi2_list)

#####PLOT 2d end######


##### PLOT 3d start###

# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")

# # Creating plot
# ax.scatter3D(logA, Omegam, Omegam_growth, c = chi2_list, s = 2, cmap=cmap, norm=matplotlib.colors.LogNorm())
# plt.title("simple 3D scatter plot")

# ax.azim = 150
# ax.elev = 15

##### PLOT 3d end###