from cobaya.likelihood import Likelihood
import numpy as np
import os
import torch
from cobaya.likelihoods.lsst_y1.nn_emulator import NNEmulator


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np
import h5py as h5


class lsst_emu_cs_lcdm(Likelihood):
    def initialize(self):


        #self.params            = config.params
        self.probe             = self.probe
        #self.n_walkers         = self.n_emcee_walkers
        self.n_pcas_baryon     = self.n_pcas_baryon
        self.baryon_pcas       = np.loadtxt(self.baryon_pca_file)
        self.emu_type          = self.emu_type
        self.dv_obs            = np.loadtxt(self.data_vector_file)[:,1]
        self.output_dims            = len(self.dv_obs)
        
        self.mask              = np.loadtxt(self.mask_file)[:,1].astype(bool)
        self.cov               = self.get_full_cov(self.cov_file)

        self.masked_inv_cov    = np.linalg.inv(self.cov[self.mask][:,self.mask])

        #print("full cov = ", self.cov)
        #print("mask = ", self.mask)
        #print("diag of cov with mask = ",self.masked_inv_cov)

        self.dv_fid            = np.loadtxt(self.data_vector_used_for_training)[:,1]
        self.dv_std            = np.sqrt(np.diagonal(self.cov))

        self.emu               = NNEmulator(self.n_dim, self.output_dims, self.dv_fid, self.dv_std)
        self.emu.load(self.emu_file)

        self.shear_calib_mask  = np.load(self.shear_calib_mask) #mask that gives 2 for cosmic shear, and 1 for gg-lensing

        self.source_ntomo      = self.source_ntomo
        if self.probe != 'cosmic_shear':
            self.lens_ntomo = self.lens_ntomo
            self.galaxy_bias_mask = np.load(self.galaxy_bias_mask) # mask that gives 2 for galaxy clustering, and 1 for gg-lensing
            self.n_fast_pars = self.n_pcas_baryon + self.source_ntomo + self.lens_ntomo
        else:
            self.n_fast_pars = self.n_pcas_baryon + self.source_ntomo
        
        #self.m_shear_prior_std = split_with_comma(self.config_args_emu['shear_calib']['prior_std'])
        
        #NO baryons for now
        #self.config_args_baryons = self.config_args_emu['baryons']
        
        if self.probe!='cosmic_shear':
            self.config_args_bias  = self.config_args_emu['galaxy_bias']
            self.bias_fid          = split_with_comma(self.config_args_bias['bias_fid'])
            self.galaxy_bias_mask  = self.galaxy_bias_mask

        if self.probe!='cosmic_shear':
            self.n_sample_dims    = self.n_dim + self.lens_ntomo + self.source_ntomo + self.n_pcas_baryon
        else:
            self.n_sample_dims    = self.n_dim + self.source_ntomo + self.n_pcas_baryon

        #super(lsst_emu_cs_lcdm, self).initialize()

    def get_requirements(self):
        return {
          "H0": None,
          "omegab": None,
          "omegam": None
        }


    def get_full_cov(self, cov_file):
        print("Getting covariance...")
        full_cov = np.loadtxt(cov_file)
        cov = np.zeros((self.output_dims, self.output_dims))
        cov_scenario = full_cov.shape[1]
        
        for line in full_cov:
            i = int(line[0])
            j = int(line[1])

            if(cov_scenario==3):
                cov_ij = line[2]
            elif(cov_scenario==10):
                cov_g_block  = line[8]
                cov_ng_block = line[9]
                cov_ij = cov_g_block + cov_ng_block

            cov[i,j] = cov_ij
            cov[j,i] = cov_ij

        return cov

    # Get the parameter vector from cobaya
    #TODO: check the order is the same as trained emulator

    def get_theta(self, **params_values):

      theta = np.array([])

      # 6 cosmological parameter for LCDM with gg-split
      logAs = self.provider.get_param("logA")
      ns = self.provider.get_param("ns")
      H0 = self.provider.get_param("H0")
      omegab = self.provider.get_param("omegab")
      omegam = self.provider.get_param("omegam")
      
      theta = np.append(theta, [logAs, ns, H0, omegab, omegam])  #NEED this order for now

      # 7 nuissance parameter emulated
      LSST_DZ_S1 = params_values['LSST_DZ_S1']
      LSST_DZ_S2 = params_values['LSST_DZ_S2']
      LSST_DZ_S3 = params_values['LSST_DZ_S3']
      LSST_DZ_S4 = params_values['LSST_DZ_S4']
      LSST_DZ_S5 = params_values['LSST_DZ_S5']

      LSST_A1_1 = params_values['LSST_A1_1']
      LSST_A1_2 = params_values['LSST_A1_2']



      theta = np.append(theta, [LSST_DZ_S1, LSST_DZ_S2, LSST_DZ_S3, LSST_DZ_S4, LSST_DZ_S5, LSST_A1_1, LSST_A1_2])  #NEED this order for now

      # 5 fast parameters don't emulate, no baryons for now
      LSST_M1 = params_values['LSST_M1']
      LSST_M2 = params_values['LSST_M2']
      LSST_M3 = params_values['LSST_M3']
      LSST_M4 = params_values['LSST_M4']
      LSST_M5 = params_values['LSST_M5']

      theta = np.append(theta, [LSST_M1, LSST_M2, LSST_M3, LSST_M4, LSST_M5])  #NEED this order for now

      if self.probe!='cosmic_shear':
        LSST_DZ_L1 = params_values['LSST_DZ_L1']
        LSST_DZ_L2 = params_values['LSST_DZ_L2']
        LSST_DZ_L3 = params_values['LSST_DZ_L3']
        LSST_DZ_L4 = params_values['LSST_DZ_L4']
        LSST_DZ_L5 = params_values['LSST_DZ_L5']

        LSST_B1_1  = params_values['LSST_B1_1']
        LSST_B1_2  = params_values['LSST_B1_2']
        LSST_B1_3  = params_values['LSST_B1_3']
        LSST_B1_4  = params_values['LSST_B1_4']
        LSST_B1_5  = params_values['LSST_B1_5']

        theta = np.append(theta, [LSST_DZ_L1, LSST_DZ_L2, LSST_DZ_L3, LSST_DZ_L4, LSST_DZ_L5])  #NEED this order for now
        theta = np.append(theta, [LSST_B1_1, LSST_B1_2, LSST_B1_3, LSST_B1_4, LSST_B1_5])  #NEED this order for now

      if self.n_pcas_baryon ==4:
        LSST_BARYON_Q1 = params_values['BARYON_Q1']
        LSST_BARYON_Q2 = params_values['BARYON_Q2']
        LSST_BARYON_Q3 = params_values['BARYON_Q3']
        LSST_BARYON_Q4 = params_values['BARYON_Q4']

        theta = np.append(theta, [LSST_BARYON_Q1, LSST_BARYON_Q2, LSST_BARYON_Q3, LSST_BARYON_Q4])  #NEED this order for now


      # #for 2D check
      # theta = np.array([logAs, ns])
      # print(theta)
      return theta

    # Get the dv from emulator
    def compute_datavector(self, theta):        
        if(self.emu_type=='nn'):
            theta = torch.Tensor(theta)
        elif(self.emu_type=='gp'):
            theta = theta[np.newaxis]

        datavector = self.emu.predict(theta)[0]


        return datavector

    # add the fast parameter part into the dv
    def get_data_vector_emu(self, theta):
        theta_emu     = theta[:-self.n_fast_pars]

        datavector = self.compute_datavector(theta_emu)

        # #For 2D check
        # print("theta_emu = ", theta_emu)
        # datavector = self.compute_datavector(theta)
        # print("datavector = ", datavector)
        # print("testing 2D", (datavector -self.dv_fid )/self.dv_fid )


        if self.probe!='cosmic_shear':
            bias_theta = theta[self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo + self.lens_ntomo):
                                  self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo)]
            datavector = self.add_bias(bias_theta, datavector)
        m_shear_theta = theta[self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo):
                              self.n_sample_dims-self.n_pcas_baryon]
        #print('dv before m_shear',datavector[0:20])
        datavector = self.add_shear_calib(m_shear_theta, datavector)
        #print('dv after m_shear',datavector[0:20])
        if(self.n_pcas_baryon > 0):
            baryon_q   = theta[-self.n_pcas_baryon:]
            datavector = self.add_baryon_q(baryon_q, datavector)
        return datavector


    def add_bias(self, bias_theta, datavector):
        for i in range(self.lens_ntomo):
            factor = (bias_theta[i] / self.bias_fid[i])**self.galaxy_bias_mask[i]
            datavector = factor * datavector
        return datavector

    def add_baryon_q(self, Q, datavector):
        for i in range(self.n_pcas_baryon):
            datavector = datavector + Q[i] * self.baryon_pcas[:,i]
        return datavector

    def add_shear_calib(self, m, datavector):
        for i in range(self.source_ntomo):
            factor = (1 + m[i])**self.shear_calib_mask[i]
            factor = factor[0:780] #for cosmic shear
            datavector = factor * datavector
        return datavector


    def logp(self, **params_values):
        theta = self.get_theta(**params_values)
        model_datavector = self.get_data_vector_emu(theta)
        #print("theta = ",theta)
        #print("dv_obs(used for evaluation) = ", self.dv_obs, 'with shape', np.shape(self.dv_obs), self.dv_obs[-10:])
        #print("emulated_dv = ", model_datavector)

        delta_dv = (model_datavector - self.dv_obs[0:780])[self.mask[0:780]]
        
        #print("delta_dv / dv_fid = ", delta_dv/self.dv_obs[0:780][self.mask[0:780]], "lendth is ", len(delta_dv))
        #print("testing.....", self.dv_obs[self.mask] @self.masked_inv_cov @ self.dv_obs[self.mask])
        logp = -0.5 * delta_dv @ self.masked_inv_cov[0:780, 0:780] @ delta_dv  
        return logp

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gain + self.bias