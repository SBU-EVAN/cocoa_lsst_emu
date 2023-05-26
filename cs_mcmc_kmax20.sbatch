#!/bin/bash

#SBATCH --job-name=shear
#SBATCH --output=cs_k20.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=9
#SBATCH --partition=extended-40core
#SBATCH --time=168:00:00

# Clear the environment from any previously loaded modules

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

cd $SLURM_SUBMIT_DIR
module purge > /dev/null 2>&1

source /gpfs/home/esaraivanov/conda/etc/profile.d/conda.sh
module load slurm
conda activate cocoatorch
source start_cocoa
module load gcc/10.2.0

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

mpirun -n ${SLURM_NTASKS} --mca btl tcp,self --bind-to core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/lsst_y1/cs_mcmc_kmax20.yaml -r