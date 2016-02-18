#!/bin/sh
# submit.sh
#SBATCH --partition=devel
#SBATCH --gres=gpu:1
#SBATCH --job-name=e1_run1
seq 101 200 >seeder
./build-restart.sh $SLURM_JOB_NODELIST $SLURM_JOB_NAME $SLURM_JOB_ID
sbatch submit-restart.sh
srun ./bluebottle -n 100
