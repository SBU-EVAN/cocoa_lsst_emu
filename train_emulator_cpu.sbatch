#!/bin/bash

#SBATCH --job-name=lsst_emu
#SBATCH --output=lsst_train_%A.out
#SBATCH -e lsst_train_%A.out.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
##SBATCH --gpus-per-task=1
#SBATCH --partition=long-40core
#SBATCH -t 48:00:00

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

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

python3 train_emulator.py \
        ./projects/lsst_y1/train_emulator_dzl.yaml \
        ./projects/lsst_y1/emulator_output/chains/train_post_T8_3x2
#        ./projects/lsst_y1/emulator_output/chains/train_post_600k_T8_-4omegam \
#        ./projects/lsst_y1/emulator_output/chains/train_post_600k_T8_+4omegam \
#        ./projects/lsst_y1/emulator_output/chains/train_post_600k_T8_+4pc1 \
#        ./projects/lsst_y1/emulator_output/chains/train_post_600k_T8_-4pc1 \
#        ./projects/lsst_y1/emulator_output/chains/train_post_600k_T8_+4sigma8 \
#        ./projects/lsst_y1/emulator_output/chains/train_post_600k_T8_-4sigma8
#        ./projects/lsst_y1/emulator_output/chains/train_post_T10_2
#./projects/lsst_y1/emulator_output/dvs_for_training_100k/train
