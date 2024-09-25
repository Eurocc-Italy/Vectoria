#!/bin/bash

#SBATCH --account=PhDLR_prod
#SBATCH --job-name=build_index
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --time 00:15:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 1 task
#SBATCH --cpus-per-task=8   # 8 cores out of 32
#SBATCH --gres=gpu:1        # 1 gpus per node out of 4
#SBATCH --mem=300000        # memory per node out of 494000MB (481GB)

##SBATCH --qos=boost_qos_dbg

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Avoid fragmentation

export TMPDIR="/leonardo_scratch/large/userinternal/aproia00"
export TMP="/leonardo_scratch/large/userinternal/aproia00"

module load profile/deeplrn
module load cineca-ai/4.3.0
echo "All modules loaded..."

source /leonardo_work/PhDLR_prod/eucc-env/bin/activate
echo "Environment loaded..."

echo "START"
python vectoria build_index --input_docs_dir /leonardo_work/PhDLR_prod/vectoria/eurocc_prod/test/data/pdf --output_index_dir /leonardo_work/PhDLR_prod/vectoria/eurocc_prod/test/data/out
#srun ...
#srun -n 1 ...
echo "END"