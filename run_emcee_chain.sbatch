#!/bin/bash

#SBATCH --job-name=dv_chain
#SBATCH --output=lsst_emu-%A_%a.out
#SBATCH -e lsst_emu-%A_%a.out.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --partition=long-28core
#SBATCH -t 48:00:00

temperature=256

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

module purge > /dev/null 2>&1
cd $SLURM_SUBMIT_DIR

source /gpfs/home/esaraivanov/conda/etc/profile.d/conda.sh
module load slurm
conda activate cocoatorch
source start_cocoa

module load gcc/10.2.0
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/home/esaraivanov/conda/lib
  
export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

echo Running emcee chain with T=$temperature

python gaussian_approx.py none 0 $temperature train_atplanck_T256

#echo calculating dvs

#$CONDA_PREFIX/bin/mpirun -n ${SLURM_NTASKS} --mca btl tcp,self --bind-to core --map-by numa:pe=${OMP_NUM_THREADS} python3 get_dv_from_chain.py ./projects/lsst_y1/dv_from_chain.yaml ./projects/lsst_y1/emulator_output/chains/gaussian_approx_0_$temperature -f