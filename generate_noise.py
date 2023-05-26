import sys,os
import numpy as np
sys.path.insert(0, os.path.abspath(".."))
from cocoa_emu import Config

# open yaml config. 
configfile = sys.argv[1]
config = Config(configfile)

if config.probe=='cosmic_shear':
    print("training for cosmic shear only")
    start=0
    stop=780
    sample_dim=12
    validation_root='./projects/lsst_y1/emulator_output/chains/vali_post_T1'
elif config.probe=='3x2pt':
    # 3x2pt is generally very difficult.
    print("trianing for 3x2pt")
    start=0
    stop=1560
    validation_root='./projects/lsst_y1/emulator_output/chains/vali_post_T1_3x2'
elif config.probe=='2x2pt':
    print("training for 2x2")
    start=780
    stop=1560
    validation_root='./projects/lsst_y1/emulator_output/chains/vali_post_T1_3x2'
else:
    print('probe not defined')
    quit()

cov = config.cov[start:stop,start:stop]
fid = config.dv_fid[start:stop]
noise_realizations = np.random.multivariate_normal(fid, cov, size=1000)
print(noise_realizations.shape)
np.savetxt('cosmicshear_noise.txt',noise_realizations)

