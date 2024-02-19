import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from cocoa_emu import cocoa_config, nn_pca_emulator
from cocoa_emu.sampling import EmuSampler
import emcee
from getdist import plots, MCSamples
import sys
# just for convinience
from datetime import datetime
from multiprocessing import Pool

print(os.getcwd())


def compute_omega_b_c(theta):
    params   = theta[:5]
    H0       = params[2]
    omegab   = params[3]
    omegam   = params[4]
    omeganh2 = (3.046/3)**(3/4)*0.06/94.1

    h = H0/100

    omegabh2 = omegab*(h**2)
    omegach2 = (omegam-omegab)*(h**2) - omeganh2


    return(omegabh2,omegach2)

# priors and likelihood
cosmo_prior_lim = np.array([[1.61, 3.91],
                       [0.87, 1.07],
                       [55, 91],
                       [0.03,0.07],#[0.009075,0.057967],#[0.03,0.07],#[0.01, 0.04],
                       [0.1, 0.9]])
ia_prior_lim = np.array([[-5., 5.],
                       [-5., 5.]])
dz_source_std   = np.array([0.005,0.002,0.002,0.003,0.002])
shear_calib_std = 0.005 * np.ones(5)

def add_shear_calib(m, datavector):
    for i in range(5):
        factor = (1 + m[i])**shear_calib_mask[i]
        datavector = factor * datavector
    return datavector

def hard_prior(theta, params_prior):
    is_lower_than_min = bool(np.sum(theta < params_prior[:,0]))
    is_higher_than_max = bool(np.sum(theta > params_prior[:,1]))
    if is_lower_than_min or is_higher_than_max:
        return -np.inf
    else:
        return 0.
    
def lnprior(theta):
    cosmo_theta = theta[:5]
    ns          = cosmo_theta[1]
    ns_prior    = 0.

    dz_source   = theta[5:10]
    ia_theta    = theta[10:12]
    shear_calib = theta[12:17]

    cosmo_prior = hard_prior(cosmo_theta, cosmo_prior_lim) + ns_prior
    ia_prior    = hard_prior(ia_theta, ia_prior_lim)
    
    dz_source_lnprior   = -0.5 * np.sum((dz_source / dz_source_std)**2)
    shear_calib_lnprior = -0.5 * np.sum((shear_calib / shear_calib_std)**2)
    
    return cosmo_prior + ia_prior + dz_source_lnprior + shear_calib_lnprior
    
def ln_lkl(theta):
    param = theta[:12]
    shear = theta[12:17]
    omb,omc = compute_omega_b_c(param)
    param[3]=omb
    param[4]=omc
    dv_cs = emu_cs.predict(torch.Tensor(param))[0]
    dv = add_shear_calib(shear,dv_cs)[mask]
    dv_diff_masked = (dv - dv_fid)
    lkl = -0.5 * dv_diff_masked @ cov_inv_masked @ dv_diff_masked
    return lkl

def lnprob(theta):
    prob=lnprior(theta) + ln_lkl(theta)
    return prob

configfile = './projects/lsst_y1/train_emulator.yaml'
config = cocoa_config(configfile)

OUTPUT_DIM=780
mask             = config.mask[:780]
#reduced_dim_idxs = np.arange(0,780)[mask]
cov_inv_masked   = config.cov_inv_masked#[reduced_dim_idxs][:,reduced_dim_idxs]
print(cov_inv_masked.shape)
#cov_inv_masked   = np.linalg.inv(config.cov[0:OUTPUT_DIM, 0:OUTPUT_DIM][mask][:,mask])
#shear_calib_mask = config.shear_calib_mask[:,:780]

# set needed parameters to initialize emulator
device=torch.device('cpu')
# torch.set_num_interop_threads(28) # Inter-op parallelism
# torch.set_num_threads(28) # Intra-op parallelism
evecs=0

#noise = np.loadtxt('cosmicshear_noise.txt')

emu_cs = nn_pca_emulator(nn.Sequential(nn.Linear(1,1)), config.dv_fid, np.ones(780), np.identity(780), evecs, device)
emu_cs.load('./projects/lsst_y1/emulator_output/models/lsst_atplanck_T128_attention')
#emu_cs.load('./projects/lsst_y1/emulator_output/models/for_tables/cosmic_shear_resnet_nlayer_1_intdim_256')
emu_cs.model.double()

# load noise realizations

N_MCMC        = 5000
N_WALKERS     = 120
NDIM_SAMPLING = 17

theta0    = np.array([3.047, 0.97, 67.66, 0.04, 0.3, 
                      0., 0., 0., 0., 0.,
                      0.5, 0.,
                      0., 0., 0., 0., 0.])

theta_std = np.array([0.01, 0.001, 0.5, 0.001, 0.002, 
                      0.002, 0.002, 0.002, 0.002, 0.002, 
                      0.1, 0.1,
                      0.005, 0.005, 0.005, 0.005, 0.005]) 

pos0 = theta0[np.newaxis] + 3. * theta_std[np.newaxis] * np.random.normal(size=(N_WALKERS, NDIM_SAMPLING))
dv_fid = config.dv_masked#[reduced_dim_idxs] #emu_cs.dv_fid.numpy()

shear_calib_mask = np.load('./external_modules/data/lsst_y1/emu_files/shear_calib_mask.npy')[:,:780]

with Pool(2*os.cpu_count()) as pool:
    emu_sampler = emcee.EnsembleSampler(N_WALKERS, NDIM_SAMPLING, lnprob, pool=pool)
    emu_sampler.run_mcmc(pos0, N_MCMC, progress=True)

names = [
    "logAs", "ns", "H0", "Omegab", "Omegam",
    "dz1","dz2","dz3","dz4","dz5",
    "IA1","IA2",
    "m1","m2","m3","m4","m5"
]
labels = [
    "\log(A_s)","n_s","H_0","\omega_{c}","\omega_c",
    "\Delta z_1","\Delta z_2","\Delta z_3","\Delta z_4","\Delta z_5"
    "IA1","IA2",
    "m^1","m^2","m^3","m^4","m^5"
]


samples = emu_sampler.chain[:,2500::10].reshape((-1,NDIM_SAMPLING))
legend_label = 'Emulator Chain'
chain = MCSamples(samples=samples,names = names, labels=labels, label=legend_label)
chain.saveAsText('test')









