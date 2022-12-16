#!/bin/bash
#SBATCH --partition kill-shared
#SBATCH --nodes 1
#SBATCH --mem 16gb
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 6
#SBATCH --time 1-00:00:00
#SBATCH --gres=gpu:2

# optional if you use come kind of virtual env:
# conda activate sar_new
time python /dir/to/ddp_tutorial/ddp_tutorial/full_ddp.py -n 1 -g 2 -v 0 -nb 1000000 -d 10 -z 1000

