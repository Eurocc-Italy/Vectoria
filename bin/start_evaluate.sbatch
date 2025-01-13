#!/bin/bash

#SBATCH --account=<HPC_ACCOUNT>
#SBATCH --job-name=build_index
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=<HPC_PARTITION>
#SBATCH --time 00:15:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 1 task
#SBATCH --cpus-per-task=8   # 8 cores out of 32
#SBATCH --gres=gpu:1        # 1 gpus per node out of 4
##SBATCH --exclusive        # reserve full node for job
                            # NOTE: gpu/cpu requirements will be ignored

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Avoid fragmentation

export TMPDIR="/path/to/scratch/dir"
export TMP="/path/to/scratch/dir"

# Load necessary modules
module load <...>
echo "All modules loaded..."

source /path/to/installdir/vectoria_env/bin/activate
echo "Environment loaded..."

TEST_SET_PATH=/path/to/installdir/vectoria/test/data/results/test_set.json
OUTPUT_DIR=/path/to/installdir/vectoria/test/data/results
CONFIG_FILE_PATH=/path/to/installdir/vectoria/etc/default/cineca_config.yaml

echo "START"
python vectoria --config $CONFIG_FILE_PATH evaluate --test-set-path $TEST_SET_PATH --output-dir $OUTPUT_DIR
echo "END"
