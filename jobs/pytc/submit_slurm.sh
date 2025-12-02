#!/bin/bash
#SBATCH --job-name=pytc
#SBATCH --time=00:30:00
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=64G
#SBATCH --gpus=a100:1
#SBATCH --array=0-15%4
#SBATCH --output=magneton/jobs/pytc/logs/%x_%A_%a.out
#SBATCH --error=magneton/jobs/pytc/logs/%x_%A_%a.err
#SBATCH --partition=gpu
module load StdEnv
export SLURM_EXPORT_ENV=ALL
source /gpfs/radev/home/zz545/miniconda3/etc/profile.d/conda.sh
conda activate pytc
cd .
python -u -m magneton.pytorch_connectomics.tools.run --config-file magneton/pytorch_connectomics/configs/SNEMI/SNEMI-Affinity-UNet.yaml --config-base magneton/pytorch_connectomics/configs/mutil/temp_config_${SLURM_ARRAY_TASK_ID}.yaml --inference --checkpoint /gpfs/marilyn/pi/kuan/shared/FIB_SEM/WORM/pretrain_checkpoint/checkpoint_110000.pth.tar
