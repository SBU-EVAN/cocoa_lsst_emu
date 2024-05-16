from .data_model import DataModel
import torch
import numpy as np

class GaussianLikelihood(DataModel):
    def __init__(self, N_DIM, config_args_io, dv_obs, cov):
        super().__init__(N_DIM, config_args_io)
        self.dv_obs = dv_obs
        self.cov = torch.Tensor(cov)
        self.inv_cov = torch.Tensor(np.linalg.inv(cov))
        self.inv_cov = np.linalg.inv(cov)
    
    def log_like(self, theta):
        model_datavector = self.compute_datavector(theta)
        delta_dv = (model_datavector - self.dv_obs)
        return -0.5 * delta_dv @ self.inv_cov @ delta_dv
