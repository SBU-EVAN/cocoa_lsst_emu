import numpy as np
import sys
import time

from typing import Mapping
import logging

from cobaya.yaml import yaml_load_file
from cobaya.model import Model
from cobaya.input import update_info
from cobaya.conventions import kinds, _timing, _params, _prior, _packages_path

def get_model(yaml_file):
    info         = yaml_load_file(yaml_file)
    updated_info = update_info(info)
    model =  Model(updated_info[_params], updated_info[kinds.likelihood],
               updated_info.get(_prior), updated_info.get(kinds.theory),
               packages_path=info.get(_packages_path), timing=updated_info.get(_timing),
               allow_renames=False, stop_at_error=info.get("stop_at_error", False))
    return model

def get_priors(params):
    lh_minmax = {}
    for x in params:
        if('prior' in params[x]):
            prior = params[x]['prior']
            if('dist' in prior):
                loc   = prior['loc']
                scale = prior['scale']
                lh_min = loc - 3. * scale
                lh_max = loc + 3. * scale
            else:
                lh_min = prior['min']
                lh_max = prior['max']
            lh_minmax[x] = {'min': lh_min, 'max': lh_max}
    return lh_minmax

def get_lhs_priors(cosmology_yaml_file, cocoa_yaml_file):
    cosmology_info      = yaml_load_file(cosmology_yaml_file)
    cosmology_params    = cosmology_info['params']

    cocoa_params      = yaml_load_file(cocoa_yaml_file)

    priors       = get_priors(cosmology_params)
    cocoa_priors = get_priors(cocoa_params)
    priors.update(cocoa_priors)
    
    labels = []
    for x in priors:
        labels.append(x)
  
    return labels, priors

def get_params_from_sample(sample, labels, lhs_prior=None):
    assert len(sample)==len(labels), "Length of the labels not equal to the length of samples"
    params = {}
    for i, label in enumerate(labels):
        if lhs_prior is None:
            param_i = sample[i]
        else:
            lhs_min = lhs_prior[label]['min']
            lhs_max = lhs_prior[label]['max']
            param_i = lhs_min + (lhs_max - lhs_min) * sample[i]
        params[label] = param_i
    return params

def get_params_list(samples, labels, lhs_prior=None):
    params_list = []
    for i in range(len(samples)):
        params = get_params_from_sample(samples[i], labels, lhs_prior)
        params_list.append(params)
    return params_list

def params_dict2array(params_dict):
    labels     = []
    array_list = []
    for x in params_dict:
        labels.append(x)
        array_list.append(params_dict[x])
    return labels, np.array(array_list)