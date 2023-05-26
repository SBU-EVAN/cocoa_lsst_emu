# Running
### Step 0: conda env. 

Use the cocoatorch conda env (need the file).

### Step 1: Generate a training dataset. 

Run

```python projects/lsst_y1/gaussian_approx.py $(shifted_param) $(shift_amount) $(temperatue) $(output_file)```
  
For no shift, run with `$(shifted_param)=none` and `$(shift_amount)=0`.

### Step 2: Generate datavectors. 

First, edit the directories inside `get_dv_from_chain.py` to be the chain created in step 1. Then run
 
```sbatch projects/lsst_y1/dv_from_chain_one.sbatch```
  
### Step 3: Train the emulator. 

First, design your architecture in `train_emulator.py`. Next, ensure the configuration `.yaml` file has a no-scalecut dataset specified. Last, run

```python projects/lsst_y1/train_emulator.py $(config.yaml) -f $(training set from step 2)```
  
If you have multiple files generated in step 2, you can use them all if they have the same prefix i.e (`chain_data_vectors_0.npy` and `chain_data_vectors_1.npy` will both be loaded IF you use `-f chain`)
 
### Step 4: Test the model

Simply edit `delta_chi2.py` to load the model from step 3 and the training set generated in step 2. Then run:

```python ./projects/lsst_y1/delta_chi2.py```

### Step 5: Run an emulated chain. 

Set the model in 'emulated_chain.py' to the one trained in step 3 and the config to a standard lsst_y1 cosmic shear config with scale cuts. Then run

`python projects/lsst_y1/emulated_chain.py`
 
