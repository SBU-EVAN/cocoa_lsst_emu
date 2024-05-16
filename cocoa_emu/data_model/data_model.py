import numpy as np
import yaml
from pyDOE import lhs

class DataModel:
    def __init__(self, N_DIM, config_args_io):
        self.N_DIM = N_DIM
        self.prior_yaml_file = config_args_io
        self.prior = Prior(N_DIM, config_args_io)
        self.set_theta0_std()
    
    def compute_datavector(self):
        pass
    
    def log_prior(self, theta):
        return self.prior.compute_log_prior(theta)
    
    def log_like(self, theta):
        pass
    
    def log_prob(self, theta):
        return self.log_like(theta) + self.log_prior(theta)
    
    def set_theta0_std(self):        
        theta0_list = []
        theta_std_list = []
        for x in self.prior.prior_args:
            prior_arg = self.prior.prior_args[x]
            prior_type = get_prior_type(prior_arg)
            if(prior_type=='gauss'):
                theta0    = prior_arg['loc']
                theta_std = prior_arg['scale']
            elif(prior_type=='flat'):
                theta0    = 0.5 * (prior_arg['min'] + prior_arg['max'])
                theta_std = 0.1 * (prior_arg['max'] - prior_arg['min'])
            theta0_list.append(theta0)
            theta_std_list.append(theta_std)
        self.theta0    = np.array(theta0_list)
        self.theta_std = np.array(theta_std_list)

    def get_emcee_start_point(self, N_WALKERS):
        return self.theta0[np.newaxis] + self.theta_std[np.newaxis] * np.random.normal(size=(N_WALKERS, self.N_DIM))       

def format_prior(yaml_file, FLAG=0):
    prior_args = {}
    with open(yaml_file, "r") as stream:
        yaml_content = yaml.safe_load(stream)
    if(FLAG==0):
        yaml_params = yaml_content['params']
    else:
        yaml_params = yaml_content
    for x in yaml_params:
        has_prior = ('prior' in yaml_params[x])    
        if(has_prior):
            prior = yaml_params[x]['prior']
            prior_args[x] = prior
    return prior_args

def get_prior_type(prior_arg):
    is_flat = not ('dist' in prior_arg)
    if(is_flat):
        return 'flat'
    return 'gauss'        
    
class Prior:
    def __init__(self, N_DIM, config_args_io):
        """
        :prior_type: list of size N_DIM. Currently support either 'flat' or 'gauss'
        :prior_args: list of dictionaries of size N_DIM.
        """
        self.cosmology_yaml_file = config_args_io['cosmology_yaml_file']
        self.cocoa_yaml_file     = config_args_io['cocoa_yaml_file']
        
        prior_args       = format_prior(self.cosmology_yaml_file)
        cocoa_prior_args = format_prior(self.cocoa_yaml_file, 1)
        
        prior_args.update(cocoa_prior_args)
            
        self.prior_args = prior_args
        self.N_DIM = N_DIM
        
    def compute_log_prior(self, theta):
        log_prior = 0.
        for theta_i, label in zip(theta, self.prior_args):
            prior_arg = self.prior_args[label]
            prior_type = get_prior_type(prior_arg)
            if(prior_type=='flat'):
                log_prior += self.flat_prior(theta_i, prior_arg)
            elif(prior_type=='gauss'):
                log_prior += self.gaussian_prior(theta_i, prior_arg)
        return log_prior
    
    def flat_prior(self, theta_i, args):
        lim_lo = args['min']
        lim_hi = args['max']
        if (theta_i < lim_lo) or (theta_i > lim_hi):
            return -np.inf
        return 0.
    
    def gaussian_prior(self, theta_i, args):
        mean = args['loc']
        std  = args['scale']
        return -0.5 * (theta_i - mean)**2 / std**2       