#!/bin/bash
#SBATCH --job-name=merge
#SBATCH --time=00:30:00
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-0
#SBATCH --output=magneton/jobs/merge/logs/%x_%A_%a.out
#SBATCH --error=magneton/jobs/merge/logs/%x_%A_%a.err
#SBATCH --partition=devel
module load StdEnv
export SLURM_EXPORT_ENV=ALL
source /gpfs/radev/home/zz545/miniconda3/etc/profile.d/conda.sh
conda activate pytc
cd .
python -m magneton.toolkit.tools.merge --config magneton/toolkit/configs/config_merge.yaml
