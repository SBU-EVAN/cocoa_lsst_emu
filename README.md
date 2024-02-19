# cocoa_lsst_y1

Code to make a Neural Network to emulate real space LSST-Y1 cosmic shear data. All of the custom NN modules are in `cocoa_emu/nn_emulator.py`

As a preliminary note, I structured this as best I can to match the CoCoA setup. Following that, the contents of this folder belong in `Cocoa/projects/lsst_y1/`. 

In addition, there are a lot of improvements to be made as far as usability goes. As of now, the code is not user-friendly. I plan to work on this after the paper.

---

## Step 0: set up the conda environment
This is compactly provided in cocoatorch.sh. Simply run

```sh
source cocoatorch.sh 
conda activate cocoatorch
```

It may take some time. Please alert me of version issues.

## Step 1: Generate the training samples
I do this via a guassian approximation on the cosmological+nuisance parameters **except** fast parameters. In my example of LSST cosmic shear, I only need the $5$ cosmological parameters along with source photo-z $\Delta_{S} z_i$ and IA parameters $A_{1},\eta_1$. I do not include shear calibration $m^i$ since they act only on the datavector. 

The gaussian approximation is done using an MCMC chain (which is probably not necessary since we know how to sample a multivariate gaussian). You can run the python script

```sh
python ./projects/lsst_y1/gaussian_approx.py 0 0 $(Temperature) $(output_file_name)
```

The first two command line args are outdated, and were used to shift the chain along which parameter (first argument) and by how many sigma (second argument). 

This code reads the parameter coviariance from the file `lsst_y1_cov.txt`. One will need to adjust the `NDIM_SAMPLING` in line `155` and the priors. 

## Step 2: Compute the datavectors associated to the training samples
This part is the first interface with cosmolike. One can run this using

```sh
$CONDA_PREFIX/bin/mpirun \
-n ${NTASKS} \
--mca btl tcp,self \
--bind-to core \
--map-by numa:pe=${OMP_NUM_THREADS} \
python3 get_dv_from_chain.py \
./projects/lsst_y1/dv_from_chain.yaml \
$shift \
$idx \
$start \
$stop \
-f 
```

Again, the `shift` can be set to `0` as it has no effect, while `start` and `stop` are used to control how many datavectors you want to compute (use `start=0`, `stop=-1` to compute the datavector for ALL parameters you generated). The code here is quite messy and I did not follow good practice especially here. The following need to be changed:

 - I hard-coded the parameters in line `56`.
 - Lines `88-91` contain the path for the sample file to load, which gets read as a getdist chain. 
 - Lines `113-114` contain the ouptut file name. 

The samples and datavectors are saved in numpy binaries. The code is trivially parallelizable in MPI. 

## Step 3: Train the emulator
Now we can train the emulator. Run
```sh
python ./projects/lsst_y1/train_emulator.py $(path_to_config) \
	-f $(train_file_prefix) 
	-n $(number_of_training_samples)
	-o $(model_output_file_name)
```
The configuration is used to read the fiducial datavector and data covariance. I coded it to read a generic CoCoA configuration file. The training file prefix is used to open the samples and datavectors in order. For example, if your samples and datavectors are saved in `train_samples_0` and `train_data_vectors_0`, you should pass the code `-f train`. 

Near the beginning of the code is where you can define your model architecture. Coded in it is the best architecture I found for my paper. In this code, the datavectors are normalized via diagonalization of the covariance. In line `255` you can adjust some hyperparameters, the batch size, learning rate, etc.

The model is saved as a state_dict to the `./projects/lsst_y1/emulator_output/models/(model_ouptut_file_name)`. 

## Step 4: Test the emulator
In the code `delta_chi2.py`, change:

- The file and suffix to load an independent set of testing datavectors and samples. 
- The model in `model_list` on line `71`

The code will use the predict function in the emulator to compute the datavector and compare it to the CoCoA result. The code will save the delta_chi2 data to the file named for the model and also tell you how the mean and median delta $\chi^2$ as well as the number of points with $\chi^2>1$ and $\chi^2>0.2$.

Run with
```sh
python delta_chi2.py
```

## Step 5: Run an MCMC
In the folder `emu_likelihoods` I have an example cobaya likelihood that uses the emulator. Structurally it is similar to `get_dv_from_chain.py`, with the exception that you must specify the model at the beginning of the code. Othewise it functions just like any cobaya likelihood. You can use it with the yaml file `emulator_chain_mcmc.yaml`.

## Other information
There is some structural comments to make.
1. The datavector normalization is done in `train_emulator.py`, while the cosmological parameter normalization is done in `cocoa_emu/nn_emulator.py`. I was hoping to make it so users never had to open `nn_emulator.py` unless they wanted to change the structure of the machine learning. I haven't found a way to really make that possible, so I would like to move all normalization to `nn_emulator.py`. A user can add a class for their own normalization methods.
2. The fact that the code saves a state dict means you must always initialize your emulator with the correct structure of the layers. Along with the model state dict is a `.h5` file that contains all the necessary information for the normalization
3. The predict function takes **input parameters, not normalized parameters** as its input and gives you the **full datavector, not the datavector in the diagonal basis**. That is, no special work needs to be done. 
4. Config reads an ordinary CoCoA config, and reads the dataset file specified in the config. The exception is the shear calibration mask I use in my emulator likelihood, as it seems something that is handled within cosmolike. I would like to automate this somehow *without* altering the CoCoA style config.











