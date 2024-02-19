from cobaya.likelihood import Likelihood
import numpy as np
import os
import sys
import torch
### Replaced with load/predict function below; be careful with normalization choicies
#from projects.lsst_y1 import cocoa_emu
sys.path.append('./')
from projects.lsst_y1.cocoa_emu.nn_emulator import nn_pca_emulator 
from projects.lsst_y1.cocoa_emu.config import cocoa_config
from projects.lsst_y1.cocoa_emu.nn_emulator import Affine,ResBlock,Better_Transformer,Better_Attention

# import importlib.util
# spec_emu = importlib.util.spec_from_file_location("nn_pca_emulator", "./projects/lsst_y1/cocoa_emu/nn_emulator.py")
# nn_pca_emulator = importlib.util.module_from_spec(spec_emu)
# sys.modules["nn_pca_emulator"] = nn_pca_emulator
# spec_emu.loader.exec_module(nn_pca_emulator)

# spec_config= importlib.util.spec_from_file_location("cocoa_config", "./projects/lsst_y1/cocoa_emu/config.py")
# cocoa_config = importlib.util.module_from_spec(spec_config)
# sys.modules["cocoa_config"] = cocoa_config
# spec_config.loader.exec_module(cocoa_config)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np
import h5py as h5

import sys
sys.path.insert(0, './projects/lsst_y1/emulator_output/models/')

class lsst_emu_cs_lcdm(Likelihood):
    def initialize(self):
        super(lsst_emu_cs_lcdm,self)
        torch.set_num_threads(1)
        self.n_pcas_baryon     = 0
        self.output_dims       = 780

        #### pick the correct configuration -- TO DO: NEED TO GET FROM COBAYA AND NOT HERE
        self.config_file = './projects/lsst_y1/train_emulator.yaml'
        # self.config_file = './projects/lsst_y1/mcmc_20sigma_shift.yaml'
        # self.config_file = './projects/lsst_y1/mcmc_-20sigma_shift.yaml'
        # self.config_file = './projects/lsst_y1/5sigma_shift.yaml'
        # self.config_file = './projects/lsst_y1/-5sigma_shift.yaml'

        #### pick the correct model 
        # self.emu_file = './projects/lsst_y1/emulator_output/T128_3000k_optimal_dict'
        # self.emu_file = './projects/lsst_y1/emulator_output/T256_3000k_optimal_dict'
        # self.emu_file = './projects/lsst_y1/emulator_output/T512_3000k_optimal_dict'
        # self.emu_file = './projects/lsst_y1/emulator_output/T512_1200k_optimal_dict'
        # self.emu_file = './projects/lsst_y1/emulator_output/T512_600k_optimal_dict'

        # self.emu_file = './projects/lsst_y1/emulator_output/T128_3000k_tanh_optimal_dict'
        # self.emu_file = './projects/lsst_y1/emulator_output/T256_3000k_tanh_optimal_dict'
        # self.emu_file = './projects/lsst_y1/emulator_output/T512_3000k_tanh_optimal_dict'
        # self.emu_file = './projects/lsst_y1/emulator_output/T512_1200k_tanh_optimal_dict'
        self.emu_file = './projects/lsst_y1/emulator_output/T512_600k_tanh_optimal_dict'

        # self.emu_file = './projects/lsst_y1/emulator_output/t512_3000k_resnet_dict'
        # self.emu_file = './projects/lsst_y1/emulator_output/t512_1200k_resnet_dict'
        ####=======================
        print(self.emu_file)

        self.config = cocoa_config(self.config_file)
        self.cov_inv_masked = self.config.cov_inv_masked
        self.dv = self.config.dv_fid
        self.mask = self.config.mask
        self.device = 'cpu'

        #### MODEL
        in_dim=12
        int_dim_res = 256
        n_channels = 32
        int_dim_trf = 1024
        out_dim = 780

        layers = []
        layers.append(nn.Linear(in_dim, int_dim_res))
        # layers.append(nn.Tanh())
        layers.append(ResBlock(int_dim_res, int_dim_res))
        layers.append(ResBlock(int_dim_res, int_dim_res))
        layers.append(ResBlock(int_dim_res, int_dim_res))
        layers.append(nn.Linear(int_dim_res, int_dim_trf))
        layers.append(Better_Attention(int_dim_trf, n_channels))
        layers.append(Better_Transformer(int_dim_trf, n_channels))
        layers.append(Better_Attention(int_dim_trf, n_channels))
        layers.append(Better_Transformer(int_dim_trf, n_channels))
        layers.append(Better_Attention(int_dim_trf, n_channels))
        layers.append(Better_Transformer(int_dim_trf, n_channels))
        layers.append(nn.Linear(int_dim_trf, out_dim))
        # layers.append(nn.Linear(int_dim_res, out_dim))
        layers.append(Affine())
        nn_model = nn.Sequential(*layers)
        ####

        self.emu = nn_pca_emulator(nn_model,[0],[0],[0],[0],self.device)
        self.emu.load(self.emu_file,state_dict=True)
        self.shear_calib_mask = np.load('./external_modules/data/lsst_y1/emu_files/shear_calib_mask.npy')[:,:780]#config.shear_calib_mask[:,:780]#self.config.shear_calib_mask
        self.n_fast_pars = 5
        self.source_ntomo = 5
        self.dv_len = 780

        #datavector = self.emu.predict(torch.Tensor([3.05,0.97,70,0.02,0.12,
        #                                                0,0,0,0,0,0.5,0.3]))[0]

    def get_requirements(self):
        return {
          "logA": None,
          "H0": None,
          "ns": None,
          "omegab": None,
          "omegam": None,
          "LSST_DZ_S1": None,
          "LSST_DZ_S2": None,
          "LSST_DZ_S3": None,
          "LSST_DZ_S4": None,
          "LSST_DZ_S5": None,
          "LSST_A1_1": None,
          "LSST_A1_2": None,
          "LSST_M1": None,
          "LSST_M2": None,
          "LSST_M3": None,
          "LSST_M4": None,
          "LSST_M5": None,
        }
        ### TODO: How to remove calling camb completely?
        ### return a None doesn't solve the problem, and cause problem of "not finding omegab", 
        ### which is weird because omegabh2 to omb is trivial..
        # return {}

    # Get the parameter vector from cobaya
    # TODO: check the order is the same as trained emulator

    def compute_omega_b_c(self,pars):
        params   = pars[:5]
        H0       = params[2]
        omegab   = params[3]
        omegam   = params[4]
        omeganh2 = (3.046/3)**(3/4)*0.06/94.1

        h = H0/100

        omegabh2 = omegab*(h**2)
        omegach2 = (omegam-omegab)*(h**2) - omeganh2

        return(omegabh2,omegach2)

    def get_theta(self, **params_values):

      theta = np.array([])

      # 6 cosmological parameter for LCDM with gg-split
      logAs = self.provider.get_param("logA")
      ns = self.provider.get_param("ns")
      H0 = self.provider.get_param("H0")
      omegab = self.provider.get_param("omegab")
      omegam = self.provider.get_param("omegam")
      
      #theta = np.append(theta, [logAs, ns, H0, omegab, omegam, omegam_growth ])  #NEED this order for now

      # 7 nuissance parameter emulated
      LSST_DZ_S1 = self.provider.get_param('LSST_DZ_S1')#params_values['LSST_DZ_S1']
      LSST_DZ_S2 = self.provider.get_param('LSST_DZ_S2')#params_values['LSST_DZ_S2']
      LSST_DZ_S3 = self.provider.get_param('LSST_DZ_S3')#params_values['LSST_DZ_S3']
      LSST_DZ_S4 = self.provider.get_param('LSST_DZ_S4')#params_values['LSST_DZ_S4']
      LSST_DZ_S5 = self.provider.get_param('LSST_DZ_S5')#params_values['LSST_DZ_S5']

      LSST_A1_1 = self.provider.get_param('LSST_A1_1')#params_values['LSST_A1_1']
      LSST_A1_2 = self.provider.get_param('LSST_A1_2')#params_values['LSST_A1_2']

      # 5 fast parameters don't emulate, no baryons for now
      LSST_M1 = self.provider.get_param('LSST_M1')#params_values['LSST_M1']
      LSST_M2 = self.provider.get_param('LSST_M2')#params_values['LSST_M2']
      LSST_M3 = self.provider.get_param('LSST_M3')#params_values['LSST_M3']
      LSST_M4 = self.provider.get_param('LSST_M4')#params_values['LSST_M4']
      LSST_M5 = self.provider.get_param('LSST_M5')#params_values['LSST_M5']

      #theta = np.append(theta, [LSST_M1, LSST_M2, LSST_M3, LSST_M4, LSST_M5])  #NEED this order for now

      return np.array([logAs,ns,H0,omegab,omegam,
                    LSST_DZ_S1,LSST_DZ_S2,LSST_DZ_S3,LSST_DZ_S4,LSST_DZ_S5,
                    LSST_A1_1,LSST_A1_2,
                    LSST_M1,LSST_M2,LSST_M3,LSST_M4,LSST_M5])

    # Get the dv from emulator
    def compute_datavector(self, theta):        
        param = np.copy(theta)
        
        omb,omc = self.compute_omega_b_c(param) # need to convert omegab,omegam to omegab,omegac
        param[3]=omb
        param[4]=omc
        
        datavector = self.emu.predict(torch.Tensor(param))[0]

        return datavector

    # add the fast parameter part into the dv
    def get_data_vector_emu(self, theta):
        theta_emu     = theta[:-self.n_fast_pars]
        m_shear_theta = theta[(len(theta)-self.n_fast_pars):]

        # print("TESTING theta_emu=", theta_emu)
        
        datavector = self.compute_datavector(theta_emu)

        # if self.probe!='cosmic_shear':
        #     bias_theta = theta[self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo + self.lens_ntomo):
        #                           self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo)]
        #     datavector = self.add_bias(bias_theta, datavector)
        
        # m_shear_theta = theta[self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo):
        #                       self.n_sample_dims-self.n_pcas_baryon]
        # print('TESTING',theta, m_shear_theta, self.n_sample_dims-(self.n_pcas_baryon + self.source_ntomo), self.n_sample_dims-self.n_pcas_baryon)
        datavector = self.add_shear_calib(m_shear_theta, datavector)
        if np.isnan(datavector).any():
            print('nan encountered with params: ',theta_emu)
        #print('dv after m_shear',datavector[0:20])
        # if(self.n_pcas_baryon > 0):
        #     baryon_q   = theta[-self.n_pcas_baryon:]
        #     #print("TESTING baryon_q=", baryon_q)
        #     datavector = self.add_baryon_q(baryon_q, datavector)
        return datavector

    def add_baryon_q(self, Q, datavector):
        for i in range(self.n_pcas_baryon):
            datavector = datavector + Q[i] * self.baryon_pcas[:,i][0:self.dv_len]
        return datavector

    def add_shear_calib(self, m, datavector):
        for i in range(self.source_ntomo):
            factor = (1 + m[i])**self.shear_calib_mask[i]
            factor = factor[0:self.dv_len] # for cosmic shear
            datavector = factor * datavector
        return datavector


    def logp(self, **params_values):
        theta = self.get_theta(**params_values)
        model_datavector = self.get_data_vector_emu(theta)
        delta_dv = (model_datavector - self.dv[0:self.dv_len])[self.mask[0:self.dv_len]]
        log_p = -0.5 * delta_dv @ self.cov_inv_masked @ delta_dv 
        return log_p


#======================================================
#=== RN+TF ============================================

# N_DIM=12
# INT_DIM  = 256
# dim_frac = 8
# OUTPUT_DIM = 780

# layers = []
# layers.append(nn.Linear(N_DIM,INT_DIM))
# layers.append(ResBlock(INT_DIM,INT_DIM))
# layers.append(ResBlock(INT_DIM,INT_DIM))
# layers.append(ResBlock(INT_DIM,INT_DIM))
# layers.append(nn.Linear(INT_DIM,1024))
# layers.append(Attention(1024,dim_frac))
# layers.append(Transformer(dim_frac,1024//dim_frac))
# layers.append(nn.Linear(1024,OUTPUT_DIM))
# layers.append(Affine())
# nn_model = nn.Sequential(*layers)



#=== ResNet ============================================

# N_DIM=12
# INT_DIM  = 256
# OUTPUT_DIM = 780

# layers = []
# layers.append(nn.Linear(N_DIM,INT_DIM))
# layers.append(nn.Tanh())
# layers.append(ResBlock(INT_DIM,INT_DIM))
# layers.append(ResBlock(INT_DIM,INT_DIM))
# layers.append(ResBlock(INT_DIM,INT_DIM))
# layers.append(nn.Linear(INT_DIM,OUTPUT_DIM))
# layers.append(Affine())

# model = nn.Sequential(*layers)



#=== MLP ===============================================



#=== ResBottle =========================================











