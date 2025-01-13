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

INPUT_DOCS_DIR=/path/to/installdir/docs # [pdf, docx]
OUTPUT_INDEX_DIR=/path/to/installdir/vectoria/test/index
OUTPUT_SUFFIX=_docs
CONFIG_FILE_PATH=/path/to/installdir/vectoria/etc/default/default_config.yaml

echo "START"
python vectoria --config $CONFIG_FILE_PATH build_index --input-docs-dir $INPUT_DOCS_DIR --output-dir $OUTPUT_INDEX_DIR --output-suffix $OUTPUT_SUFFIX
echo "END"
