from .gaussian_lkl import GaussianLikelihood
import numpy as np
import torch

class LSST_3x2(GaussianLikelihood):
    def __init__(self, N_DIM, config_args_io, cov_path, dv_path, scalecut_mask_path):
        self.cov_path = cov_path
        self.dv_path  = dv_path
        self.dv_fid_path = self.dv_path 
        self.scalecut_mask_path = scalecut_mask_path
        self.mask_3x2 = np.loadtxt(self.scalecut_mask_path)[:,1].astype(bool)

        self.bias_fid = np.array([1.24, 1.36, 1.47, 1.60, 1.76])
        self.bias_mask = np.load('data/bias_mask.npy')
        self.shear_calib_mask = np.load('data/shear_calib_mask.npy')
        self.baryon_pca = np.loadtxt('data/pca.txt')

        cov      = self.get_full_cov()
        masked_cov = cov[self.mask_3x2][:,self.mask_3x2]
        self.masked_inv_cov = np.linalg.inv(masked_cov)
        dv_fid   = self.get_datavector(self.dv_fid_path)
        dv_obs   = self.get_datavector(self.dv_path)        
        
        super().__init__(N_DIM, config_args_io, dv_obs, masked_cov)
       
        self.dv_obs = dv_obs
        self.dv_fid = dv_fid
        self.dv_std = np.sqrt(np.diagonal(cov))    

    def compute_datavector(self, theta):
        if(self.emu_type=='nn'):
            theta = torch.Tensor(theta)
        elif(self.emu_type=='gp'):
            theta = theta[np.newaxis]
        datavector = self.emu.predict(theta)[0]        
        return datavector
    
    def log_prior(self, theta):
        cosmo_ia_dz_theta = theta[:17]
        bias        = theta[17:22]
        shear_calib = theta[22:27]
        baryon_q    = theta[27:29]    
        log_prior = self.prior.compute_log_prior(cosmo_ia_dz_theta)
        for b in bias:
            log_prior += self.prior.flat_prior(b, {'min': 0.8, 'max': 3.0})
        for m in shear_calib:
            log_prior += self.prior.gaussian_prior(m, {'loc': 0., 'scale': 0.005})
        for q in baryon_q:
            log_prior += self.prior.flat_prior(q, {'min': -3., 'max': 12.})
        return log_prior
    
    def get_full_cov(self):
        full_cov = np.loadtxt(self.cov_path)
        lsst_y1_cov = np.zeros((1560, 1560))
        for line in full_cov:
            i = int(line[0])
            j = int(line[1])
            
            cov_g_block  = line[-2]
            cov_ng_block = line[-1]
            
            cov_ij = cov_g_block + cov_ng_block
            
            lsst_y1_cov[i,j] = cov_ij
            lsst_y1_cov[j,i] = cov_ij

        return lsst_y1_cov

    def get_datavector(self, dv_path):
        lsst_y1_datavector = np.loadtxt(dv_path)[:,1]
        return lsst_y1_datavector
