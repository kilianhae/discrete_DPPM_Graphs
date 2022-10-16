#!/bin/bash -v
#SBATCH --job-name=my_job
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --account=tik-highmem
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL                           # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=slurm_log/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=slurm_log/%j.err                  # where to store error messages SBATCH --constraint='tesla_v100|geforce_rtx_3090'

/bin/echo Running on host: `hostname`
/bin/echo In directory: `/itet-stor/khaefeli/net_scratch/khaefeli_graph_ddpm/GraphScoreMatching_ddpm_discrete/GraphScoreMatching_ppgn`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID

python ppgn_simple.py -c ./config/train_com_small_ddpm.yaml
exit 0;