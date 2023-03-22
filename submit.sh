#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=PFC
#SBATCH --account=phph004901

#export PATH="/user/home/km14740/scratch/miniconda3/envs/preprocess38/bin:$PATH"
#export PATH="/user/home/km14740/scratch/miniconda3/bin:$PATH"
#export PATH="/mnt/storage/scratch/km14740/miniconda3/bin:$PATH"
#export PATH="/mnt/storage/scratch/km14740/miniconda3/envs/preprocess38/lib/python3.8/site-packages:$PATH"
#export PATH="/projects/Neural_networks/Emma_data_2:$PATH"
#export PATH="/user/home/km14740/scratch/new_preprocessing:$PATH"
#echo "Activating environment..."
#source /user/home/km14740/scratch/miniconda3/bin/activate preprocess38
source /mnt/storage/scratch/km14740/miniconda3/bin/activate preprocess38
echo $CONDA_PREFIX

#module add languages/anaconda3/2020.02-tflow-2.2.0
module add apps/matlab/2021a
#module load libs/cudnn/8.2.4.15-cuda-11.4.2

cd ${SLURM_SUBMIT_DIR}

echo Start Time: $(date)
# Execute code
echo $CONDA_PREFIX
python test_grid.py --name E --br PFC
echo End Time: $(date)
