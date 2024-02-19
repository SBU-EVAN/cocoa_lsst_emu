import sys,os
from mpi4py import MPI
import numpy as np
import torch
import getdist
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

from cocoa_emu import cocoa_config, get_lhs_params_list, get_params_list, CocoaModel
from cocoa_emu.sampling import EmuSampler

starttime = datetime.now()
# MPI Setup
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print(sys.argv)

# config
configfile = sys.argv[1]
start = int(sys.argv[4])
stop  = int(sys.argv[5])
print(start,stop)

config = cocoa_config(configfile)

# For no shift, set:
# shift='none'
# idx=0 
shift=sys.argv[2]
idx = sys.argv[3]

print(config.running_params)

# ============= samples from posterior =========
def get_samples_from_posterior(file_name,):
    posterior_params = getdist.mcsamples.loadMCSamples(file_name,no_cache=True)
    # if len(posterior_params.samples)<stop:
    #     stop = len(posterior_params)
    print('samples shape: {}'.format(posterior_params.samples[start:stop,:17].shape)) #12 for training
    return posterior_params.samples[start:stop,:17]

# ================== Calculate data vectors ==========================
cocoa_model = CocoaModel(configfile, config.likelihood)

def get_local_data_vector_list(params_list, rank):
    train_params_list      = []
    train_data_vector_list = []
    N_samples = len(params_list)
    N_local   = N_samples // size
    count = 0    
    for i in range(rank * N_local, (rank + 1) * N_local):
        params_arr  = np.array(list(params_list[i].values()))
        params_list_debug = {k:params_list[i][k] for k in ('logA', 'ns', 'H0', 'omegabh2', 'omegach2', 'LSST_DZ_S1', 'LSST_DZ_S2', 'LSST_DZ_S3', 'LSST_DZ_S4', 'LSST_DZ_S5', 'LSST_A1_1', 'LSST_A1_2') if k in params_list[i]} #ES
        data_vector = cocoa_model.calculate_data_vector(params_list_debug)
        train_params_list.append(params_arr)
        train_data_vector_list.append(data_vector)
        if rank==0:
            count += 1
        if rank==0 and count % 50 == 0:
            print("calculation progress, count = ", count)
    return train_params_list, train_data_vector_list

##should implement a BLOCK here for MPI safety
def get_data_vectors(params_list, comm, rank):
    print('{} is starting...'.format(rank))
    local_params_list, local_data_vector_list = get_local_data_vector_list(params_list, rank)
    print("{} is done!".format(rank))
    comm.Barrier()
    if rank!=0:
        comm.send([local_params_list, local_data_vector_list], dest=0)
        train_params       = None
        train_data_vectors = None
    else:
        data_vector_list = local_data_vector_list
        params_list      = local_params_list
        for source in range(1,size):
            new_params_list, new_data_vector_list = comm.recv(source=source)
            data_vector_list = data_vector_list + new_data_vector_list
            params_list      = params_list + new_params_list
        train_params       = np.vstack(params_list)    
        train_data_vectors = np.vstack(data_vector_list)        
    return train_params, train_data_vectors

#here is the root directory to open chains
root = '/gpfs/projects/MirandaGroup/evan/cocoa2/Cocoa/projects/lsst_y1/emulator_output/chains/'
if(rank==0):
    # need to change this to your training samples, Reads as a getdist chain
    posterior_params = get_samples_from_posterior(root+'gaussian_t512_0') 
    print("testing:", np.shape(posterior_params))
else:
    posterior_params = None

posterior_params = comm.bcast(posterior_params, root=0)
print("Calculating data vectors from posterior")

params_list = get_params_list(posterior_params, config.running_params)#config.param_labels)        
train_samples, train_data_vectors = get_data_vectors(params_list, comm, rank)    
    
if(rank==0):
    print("checking train sample shape: ", np.shape(train_samples))
    print("checking dv set shape: ", np.shape(train_data_vectors))
    # ================== Chi_sq cut ==========================
    print("not applying chi2 cut")
    try:
        os.makedirs(config.savedir)
    except FileExistsError:
        pass
    if(config.save_train_data):
        # set your save names
        np.save(config.savedir + '/train_t512_data_vectors_0.npy', train_data_vectors)
        np.save(config.savedir + '/train_t512_samples_0.npy', train_samples)

print("DONE") 
print('total runtime =',datetime.now()-starttime)
MPI.Finalize