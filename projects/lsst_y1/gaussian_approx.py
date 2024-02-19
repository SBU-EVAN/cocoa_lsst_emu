#!/usr/bin/env python

import numpy as np
import sys
import time
import emcee
import time
from numpy import linalg
import scipy
import getdist
import multiprocessing as mp

# !!!  Make sure to change your directories !!! #
shifted_param=sys.argv[1]
sigma=int(sys.argv[2])
T=int(sys.argv[3])
write_file = sys.argv[4]

print('Running chain with T={} and {} shifted {} sigma'.format(T,shifted_param,sigma))
write_dir_root = '/gpfs/projects/MirandaGroup/evan/cocoa2/Cocoa/projects/lsst_y1/emulator_output/chains/'
write_dir = write_dir_root+write_file

# reparameterize
# converts logAs, omb, omc to omm, omb, and sigma8
def emu_to_params(theta,means=None):
    logAs = theta[:,0]
    ns    = theta[:,1]
    H0    = theta[:,2]
    ombh2 = theta[:,3]
    omch2 = theta[:,4]
    
    omnh2 = (3.046/3)**(3/4)*0.06/94.1
    
    h = H0/100
    As = np.exp(logAs)/(10**10)
    
    omb = ombh2/(h**2)
    omc = omch2/(h**2)
    omn = omnh2/(h**2)
    
    omm = omb+omc+omn
    ommh2 = omm*(h**2)
    
    sigma_8 = (As/3.135e-9)**(1/2) * \
              (ombh2/0.024)**(-0.272) * \
              (ommh2/0.14)**(0.513) * \
              (3.123*h)**((ns-1)/2) * \
              (h/0.72)**(0.698) * \
              (omm/0.27)**(0.236) * \
              (1-0.014)
        
    return np.transpose(np.array([sigma_8,ns,H0,omb,omm]))
                
# converts sigma8,omm,omb to ombh2, omch2, logAs
def params_to_emu(theta):
    sigma8 = theta[:,0]
    ns = theta[:,1]
    H0 = theta[:,2]
    ob = theta[:,3]
    om = theta[:,4]
    
    h = H0/100
    omnh2 = (3.046/3)**(3/4)*0.06/94.1
    on = omnh2/(h**2)
    
    oc = om-ob-on
    obh2 = ob*(h**2)
    och2 = oc*(h**2)
    omh2 = om*(h**2)
    
    step =  (sigma8/(1-0.014)) * \
            (obh2/0.024)**(0.272) * \
            (omh2/0.14)**(-0.513) * \
            (3.123*h)**(-(ns-1)/2) * \
            (h/0.72)**(-0.698) * \
            (om/0.27)**(-0.236)
    As = (step**2)*3.135e-9
    logAs = np.log(As*(10**10))
    
    return np.transpose(np.array([logAs,ns,H0,obh2,och2]))

### Prior and Liklihood
cosmo_prior_lim = np.array([[1.61, 3.91],
                       [0.87, 1.07],
                       [55, 91],
                       [0.01, 0.04],
                       [0.001, 0.99]])

ia_prior_lim = np.array([[-7., 7.],
                       [-7., 7.]])

bias_prior_lim = np.array([[0.8, 3.],
                       [0.8, 3.],
                       [0.8, 3.],
                       [0.8, 3.],
                       [0.8, 3.]])

baryon_prior_lim = np.array([[-3., 12.],
                             [-2.5, 2.5]])

baryon_prior_lim = 3. * baryon_prior_lim 

dz_source_std   = 0.002 * np.ones(5) * T * 100
dz_lens_std     = 0.005 * np.ones(5) * T * 100
shear_calib_std = 0.005 * np.ones(5) * T * 100

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
    #dz_lens     = theta[12:17]
    #bias        = theta[17:22]
    #shear_calib = theta[22:27]
    #baryon_q    = theta[27:]
    
    cosmo_prior = hard_prior(cosmo_theta, cosmo_prior_lim) + ns_prior
    ia_prior    = hard_prior(ia_theta, ia_prior_lim)
    #bias_prior  = hard_prior(bias, bias_prior_lim)
    #baryon_prior = hard_prior(baryon_q, baryon_prior_lim)
    
    dz_source_lnprior   = -0.5 * np.sum((dz_source / dz_source_std)**2)
    #dz_lens_lnprior     = -0.5 * np.sum((dz_lens / dz_lens_std)**2)
    #shear_calib_lnprior = -0.5 * np.sum((shear_calib / shear_calib_std)**2)
    
    return cosmo_prior + ia_prior + dz_source_lnprior #+ dz_lens_lnprior + \
            #shear_calib_lnprior + bias_prior + baryon_prior
    
def ln_lkl(theta):
    diff = theta-means
    #print('params: ',theta[:5])
    #print('means: ',means[:5])
    #print('cov:',param_cov[:5,:5])
    #print('diff: ',diff[:5])
    lkl = (-0.5/T) * (diff @ inv_cov @ np.transpose(diff))
    return lkl

def lnprob(theta):
    prob=lnprior(theta) + ln_lkl(theta)
    return prob

N_MCMC        = 500000
N_WALKERS     = 120
NDIM_SAMPLING = 12

fiducial  = np.array([[3.047, 0.9665, 67.66, 0.02242, 0.11933]])
# convert the fiducial to simga8,omm
fiducial_re = emu_to_params(fiducial)[0]
#shift
if shifted_param=='sigma8':
    idx=0
    stdev=np.sqrt(0.00017200530348884742)
elif shifted_param=='omegam':
    idx=4
    stdev=np.sqrt(8.021843990167796e-05)
elif shifted_param=='pc1':
    idx = [1,2,3]
    stdev=0 # for now
else:
    idx=0
    stdev=0

#shift the fiducial
print(fiducial_re)
fiducial_re[idx]+=sigma*stdev
print(fiducial_re)
fiducial = params_to_emu(np.array([fiducial_re]))[0]
mean=np.array(fiducial)

theta0    = np.append(mean,[
                      0., 0., 0., 0., 0.,
                      0.5, 0.])#,
                      #0., 0., 0., 0., 0.,
                      #1.24, 1.36, 1.47, 1.60, 1.76,
                      #0., 0., 0., 0., 0.,
                      #0., 0.])
mean = theta0

theta_std = np.array([0.01, 0.001, 0.1, 0.001, 0.002, 
                      0.002, 0.002, 0.002, 0.002, 0.002, 
                      0.1, 0.1])#,
                      #0.005, 0.005, 0.005, 0.005, 0.005, 
                      #0.03, 0.03, 0.03, 0.03, 0.03,
                      #0.005, 0.005, 0.005, 0.005, 0.005, 
                      #0.1, 0.1]) 

mean  = np.array(mean)#np.loadtxt('lsst_y1_fid.txt')
means = np.zeros(NDIM_SAMPLING)
param_cov = np.loadtxt('./projects/lsst_y1/lsst_y1_cov.txt')[:NDIM_SAMPLING,:NDIM_SAMPLING]
#print(mean[:5])
#print(param_cov[:5,:5])

# random shift if on
# if shift:
#     print('shifting...')
#     while bool(np.any(np.sum(means[:5] < cosmo_prior_lim[:,0]))) or bool(np.sum(means[:5] > cosmo_prior_lim[:,1])):
#         shift_vec = np.random.multivariate_normal(means, param_cov*2, check_valid='warn', tol=1e-8)
#         means = mean + shift_vec
# else:
means = mean
    
print(means[:5])
# random rotation if on
# if rotations:
#     print('rotating...')
#     rot_mat = scipy.stats.special_ortho_group.rvs(len(theta0))
#     print(np.linalg.det(rot_mat))
#     param_cov = np.transpose(rot_mat) @ param_cov @ rot_mat

inv_cov = np.linalg.inv(param_cov)

# Starting position of the emcee chain
pos0 = theta0[np.newaxis] + 3. * theta_std[np.newaxis] * np.random.normal(size=(N_WALKERS, NDIM_SAMPLING))

# parallel sampling
n_cpus = mp.cpu_count()
print('n_cpus = {}'.format(n_cpus))

#write
names = ['logA', 'ns', 'H0', 'omegab', 'omegac',
         'dz_source1','dz_source2','dz_source3','dz_source4','dz_source5',
         'IA1','IA2']#,
         #'dz_lens1','dz_lens2','dz_lens3','dz_lens4','dz_lens5',
         #'bias1','bias2','bias3','bias4','bias5',
         #'shear_calib1','shear_calib2','shear_calib3','shear_calib4','shear_calib5',
         #'baryon_q1','baryon_q2'
        #]
labels = names

# Do the sampling
with mp.Pool(2*n_cpus) as pool:
    sampler = emcee.EnsembleSampler(N_WALKERS, NDIM_SAMPLING, lnprob, pool=pool)
    sampler.run_mcmc(pos0, N_MCMC, progress=True)

samples = sampler.chain.reshape((-1,NDIM_SAMPLING))
mcsamples = getdist.mcsamples.MCSamples(samples=samples, names=names, labels=labels)#, ranges=cosmo_prior_lim)
mcsamples.removeBurn(0.5)
mcsamples.thin(50)
mcsamples.saveAsText(write_dir+'_0')

# To avoid high memory usage, use more samplers
# with mp.Pool(n_cpus) as pool:
#     sampler = emcee.EnsembleSampler(N_WALKERS, NDIM_SAMPLING, lnprob, pool=pool)
#     sampler.run_mcmc(pos0, N_MCMC, progress=True)
# samples = sampler.chain.reshape((-1,NDIM_SAMPLING))
# mcsamples = getdist.mcsamples.MCSamples(samples=samples, names=names, labels=labels)#, ranges=cosmo_prior_lim)
# mcsamples.removeBurn(0.5)
# mcsamples.thin(50)
# mcsamples.saveAsText(write_dir+'_1')

# with mp.Pool(n_cpus) as pool:
#     sampler = emcee.EnsembleSampler(N_WALKERS, NDIM_SAMPLING, lnprob, pool=pool)
#     sampler.run_mcmc(pos0, N_MCMC, progress=True)
# samples = sampler.chain.reshape((-1,NDIM_SAMPLING))
# mcsamples = getdist.mcsamples.MCSamples(samples=samples, names=names, labels=labels)#, ranges=cosmo_prior_lim)
# mcsamples.removeBurn(0.5)
# mcsamples.thin(50)
# mcsamples.saveAsText(write_dir+'_2')

print('Chains written to: '+write_dir)

