from subprocess import Popen
from os import mkdir
import os
import subprocess
import yaml
import copy
import numpy as np
import datetime
import time
import itertools

# The script will run a gridsearch on all the chosen parameters so try not to specify multiple values for many different parameterers. 
# If you wish to test multiple values for a parameter then simple specify them as a list of multiple elements

# Specify how many noiselevels we should use during training
noise_nums = [32]

# Not used for edp-gnn model, can just leave it like that
networksizes = [(6,32)]

# Specify which models to test on from the following: ["planar_30_200_pkl","planar_60_200_pkl","planar_90_200_pkl","trees_30_200_pkl","trees_60_200_pkl","trees_90_200_pkl","sbm_27_200_pkl","ego_18_small","community_20_small"]
datasets = [ "ego_18_small"]

# Choose the batchsize for training
batchsizes = [32]

## LEAVE THESE VALUES as these are deprecated and some functionality may produce an error if changed
weighted_losses = [True]

# choose random seeds to use
seeds = [1234]

# Chose model name (default is ppgn for normal ppgn implementation)
modelname = "edp"

# LEAVE THESE VALUES as these are deprecated and some functionality may produce an error if changed
weighted_losses = [True]
noisetypes = ["switched"]

# This creates new directories for the configs of our runs, for the slurm_job numbers, for the slurm_scripts to run based on the model parameters chosen and based on the timestamp
testdir = f"consec_edp_{datetime.datetime.now().day}.{datetime.datetime.now().month}_{datetime.datetime.now().hour}:{datetime.datetime.now().minute}"
os.system(f"mkdir config/gridsearch/{testdir}")
os.system(f"mkdir scripts/gridsearch/{testdir}")
os.system(f"mkdir gridsearch/{testdir}")

with open('config/gridsearch_edp_final.yaml') as f:
    data_base = yaml.load(f,Loader=yaml.FullLoader)
    for batchsize, dataset, networksize, noisetype, weighted_loss, seed in itertools.product(batchsizes, datasets, networksizes, noisetypes, weighted_losses,seeds):
        data = copy.copy(data_base)
        if "community" in dataset:
            data["dataset"]["dataset_size"] = 100
        data["num_layers"] = networksize[0]
        data["train"]["batch_size"] = batchsize
        data["hidden"] = networksize[1]
        data["hidden_final"] = networksize[1]
        data["dataset"]["max_node_num"] = int(dataset[dataset.find("_")+1:dataset.find("_")+3])
        data["dataset"]["name"] = dataset
        data["num_levels"] = noise_nums
        data["noisetype"] = noisetype
        data["weighted_loss"] = weighted_loss
        data["model"]["name"] = "edp-gnn"
        data["seed"] = seed
        data["train"]["sigmas"] = np.linspace(0,0.5,noise_nums[0]+1).tolist()
        data["model"]["models"]["model_1"]["name"] = "gin"
        with open(f'config/gridsearch/{testdir}/gridsearch_edp_consec_{dataset}_{networksize[0]},{networksize[1]}_{len(noise_nums)}_{batchsize}_{noisetype}_{weighted_loss}_{seed}.yaml',"w+") as g:
            yaml.dump(data, g)
        commandstring = f"python3 edp_simple.py -c config/gridsearch/{testdir}/gridsearch_edp_consec_{dataset}_{networksize[0]},{networksize[1]}_{len(noise_nums)}_{batchsize}_{noisetype}_{weighted_loss}_{seed}.yaml"

        with open('scripts/gridsearch.sh','r') as firstfile, open(f"scripts/gridsearch/{testdir}/gridsearch_edp_consec_{dataset}_{networksize[0]},{networksize[1]}_{len(noise_nums)}_{batchsize}_{noisetype}_{weighted_loss}_{seed}.sh",'a+') as secondfile:
            for line in firstfile:
                secondfile.write(line)
            secondfile.write(f'{commandstring}\n')
            secondfile.write("exit 0;")

        ##ATTENTION this command must be specifically changed to your slurm service
        out = subprocess.check_output(f"sbatch scripts/gridsearch/{testdir}/gridsearch_edp_consec_{dataset}_{networksize[0]},{networksize[1]}_{len(noise_nums)}_{batchsize}_{noisetype}_{weighted_loss}_{seed}.sh",shell=True)
        index = str(out).find("\n")
        jobnumber = int(out[index-6:index])
        with open(f"gridsearch/{testdir}/gridsearch_edp_consec_{dataset}_{networksize[0]},{networksize[1]}_{len(noise_nums)}_{batchsize}_{noisetype}_{weighted_loss}_{seed}.txt","w+") as idfile:
            idfile.write(f"{jobnumber}")
                                







#os.system("sbatch -C gpu -v scripts/gridsearch_consec_sbm.sh")
   
