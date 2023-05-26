import numpy as np
import getdist
from getdist import plots
import matplotlib.pyplot as plt

import torch
from torch import nn
from cocoa_emu import cocoa_config, nn_pca_emulator
from cocoa_emu.sampling import EmuSampler

# function to convert omega_m and sigma8
def compute_omegam_sigma8(theta):
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
        
    return (omm,sigma_8,As)

#parameter ranges
ranges={'logAs':(1.61,3.91),
        'ns':(0.87,1.07),
        'H0':(55,91),
        'omegab':(0.03,0.07),
        'omegam':(0.1,0.9)
}

# debug file to plot samples used for training in a triangle plot

base_path = './projects/lsst_y1/emulator_output/'
names = ['logA','ns','H0','omegab','omegam','DZS1','DZS2','DZS3','DZS4','DZS5','IA1','IA2','M1','M2','M3','M4','M5','chi2']
label = names
# names = ['omegam','sigma8']
# label = ['\Omega_m','\sigma_8']
# idxs = [26,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] #sigma8 omegam
# idxs = [2,3,4,21,22,7,8,9,10,11,12,13,14,15,16,17,18,-1]
idxs = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,-1]

chain1 = np.loadtxt('./projects/lsst_y1/chains/lsst_at_planck_kmax5.1.txt')[:,idxs]#,4,5,6]]#[22,27]]
chain2 = np.loadtxt('./projects/lsst_y1/chains/lsst_at_planck_kmax5.2.txt')[:,idxs]#,4,5,6]]#[22,27]]
chain3 = np.loadtxt('./projects/lsst_y1/chains/lsst_at_planck_kmax5.3.txt')[:,idxs]#,4,5,6]]#[22,27]]
chain4 = np.loadtxt('./projects/lsst_y1/chains/lsst_at_planck_kmax5.4.txt')[:,idxs]#,4,5,6]]#[22,27]]

len1 = len(chain1)
len2 = len(chain2)
len3 = len(chain3)
len4 = len(chain4)

#burn in and thin
thin_frac = 10
burn_in_frac = 0.5

chain1 = chain1[int(0.5*len1)::thin_frac]
chain2 = chain2[int(0.5*len2)::thin_frac]
chain3 = chain3[int(0.5*len3)::thin_frac]
chain4 = chain4[int(0.5*len4)::thin_frac]

chain = np.vstack((chain1,chain2,chain3,chain4))
print(chain.shape)
mcmc1  = getdist.mcsamples.MCSamples(samples=chain,names=names,labels=label,label='Cocoa',ranges=ranges)
omm,sigma8,As = compute_omegam_sigma8(mcmc1.samples)

#mcmc1.addDerived(omm,name='omegam',label='\Omega_m')
mcmc1.addDerived(sigma8,name='sigma8',label='\sigma_8')
mcmc1.addDerived(As,name='As',label='A_s')

# open and convert emulator chain
chain2 = getdist.mcsamples.loadMCSamples('./lsst_at_planck_test_mobo',no_cache=True)
chain2.setParamNames(['logA','ns','H0','omegab','omegam','DZS1','DZS2','DZS3','DZS4','DZS5','IA1','IA2','M1','M2','M3','M4','M5'])
print(chain2.getParamSampleDict(0))
omm,sigma8,As = compute_omegam_sigma8(chain2.samples)

#chain2.addDerived(omm,name='omegam',label='\Omega_m')
chain2.addDerived(sigma8,name='sigma8',label='\sigma_8')
chain2.addDerived(As,name='As',label='A_s')

g = plots.get_subplot_plotter()
g.settings.num_plot_contours = 2
g.triangle_plot([mcmc1,chain2],
               filled=True,
               params=['As','omegam','omegab','H0','ns','DZS1','DZS2','DZS3','DZS4','DZS5','IA1','IA2','M1','M2','M3','M4','M5'])
g.export('cocoa_vs_emu_posteriors.pdf')
