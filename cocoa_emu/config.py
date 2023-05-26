import yaml
import numpy as np
import os

class cocoa_config:
    '''
    datafile:             .datafile listed in config
    likelihood:           name of the likelihood used
    dv_fid:               the fiducial cosmology, where the likelihood is centered
    mask:                 mask from file
    cov:                  full cov
    inv_cov:              unmasked inv covariance
    inv_cov_masked:       masked inverse covariance
    dv_masked:            masked fiducial datavector
    running_names:        names of cocoa sampling params
    running_names_latex:  latex labels for sampling params
    # shear_calib_mask:   mask for shear calibration # NOT IMPLEMENTED
    # bias_mask:          mask for galaxy bias       # NOT IMPLEMENTED
    '''

    def __init__(self, config_file):
        with open(config_file,'r') as stream:
            config_args = yaml.safe_load(stream)

        config_lkl    = config_args['likelihood'] # dataset file with dv_fid, mask, etc.
        config_params = config_args['params']     # list of params, used to create arrays and dicts

        self.load_lkl(config_lkl)
        self.load_params(config_params)

        print('loaded config')

    def load_lkl(self, lkl_args):
        '''
        setup the lkl from lkl_args in the yaml
        also open .dataset file (default is LSST_Y1.dataset, but this is not a good choice for most applications)
        '''

        self.likelihood = list(lkl_args.keys())[0]
        _lkl = lkl_args[self.likelihood] # get for dataset file

        path = _lkl['path']
        
        try:
            self.data_file = _lkl['data_file']
        except:
            print('Argument not found in configuration file: "data_file"')
            print('  > using "LSST_Y1.dataset" (NOT recommended for training)')
            self.data_file = 'LSST_Y1.dataset'
        
        file = path+'/'+self.data_file
        data = open(file, 'r')

        for line in data.readlines():
            split = line.split()
            # need: dv_fid, cov, mask.
            if(split[0] == 'data_file'):
                dv_fid_file = split[2]
            elif(split[0] == 'cov_file'):
                cov_file = split[2]
            elif(split[0] == 'mask_file'):
                mask_file = split[2]
        
        self.dv_fid = self.get_vec_from_file(path+'/'+dv_fid_file)
        self.mask   = self.get_vec_from_file(path+'/'+mask_file).astype(bool)

        self.dv_masked = self.dv_fid[self.mask]

        self.get_cov(path+'/'+cov_file,self.mask) # function creates cov, inv_cov, masked_inv_cov
        print('datafile read complete')

        return
            
    def get_vec_from_file(self,file):
        vec  = np.loadtxt(file)
        idxs = np.argsort(vec[:,0]) # first column is index, second column is entry

        return(vec[:,1][idxs])
    
    def get_cov(self,file,mask):
        full_cov = np.loadtxt(file)
        cov_scenario = full_cov.shape[1]
        size = len(mask)

        self.cov            = np.zeros((size,size))
        self.cov_inv_masked = np.zeros((size,size))

        for line in full_cov:
            i = int(line[0])
            j = int(line[1])

            if(cov_scenario == 3):
                cov_ij = line[2]
            elif(cov_scenario == 4):
                cov_ij = line[2]+line[3]
            elif(cov_scenario == 10):
                cov_ij = line[8]+line[9]

            self.cov[i,j] = cov_ij
            self.cov[j,i] = cov_ij

        for i in range(size):
            for j in range(i,size):
                if(i!=j):
                    mask_row    = mask[i]
                    mask_column = mask[j]

                    self.cov_inv_masked[i,j] = self.cov[i,j] * mask_row * mask_column
                    self.cov_inv_masked[j,i] = self.cov_inv_masked[i,j]
                else:
                    self.cov_inv_masked[i,j] = self.cov[i,j]

        self.cov_inv = np.linalg.inv(self.cov)
        self.cov_inv_masked = np.linalg.inv(self.cov_inv_masked)[mask][:,mask]

        return

    def load_params(self, param_args):
        params_list = param_args.keys()

        self.running_params       = []
        self.running_params_latex = []

        for param in params_list:
            keys = param_args[param].keys()
            if('value' not in keys and 'derived' not in keys and len(keys)>1):
                self.running_params.append(param)
                self.running_params_latex.append(param_args[param]['latex'])

        return