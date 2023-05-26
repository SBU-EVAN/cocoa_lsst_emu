import numpy as np

n=0

sample = np.load('./emulator_output/chains/cocoa_chain_training_lkl_samples.npy')[n]
datavc = np.load('./emulator_output/chains/cocoa_chain_training_lkl_dvs.npy')[n]

np.savetxt('_debug_sample.txt',sample)
np.savetxt('_debug_datavc.txt',datavc)