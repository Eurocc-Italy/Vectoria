#!/bin/bash

#SBATCH --account=cin_staff
#SBATCH --job-name=build_index
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --time 00:15:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 1 task
##SBATCH --cpus-per-task=8   # 8 cores out of 32
#SBATCH --gres=gpu:1        # 1 gpus per node out of 4
##SBATCH --mem=300000        # memory per node out of 494000MB (481GB)
#SBATCH --exclusive

##SBATCH --qos=boost_qos_dbg

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Avoid fragmentation

export TMPDIR="/leonardo_scratch/large/userinternal/lbabetto"
export TMP="/leonardo_scratch/large/userinternal/lbabetto"

# module load profile/deeplrn
# module load cineca-ai/4.3.0
echo "All modules loaded..."

source /leonardo_scratch/fast/EUCC_staff_3/lbabetto/vectoria-work/clean-env/bin/activate
# source /leonardo_scratch/fast/EUCC_staff_3/lbabetto/vectoria-work/vectoria-env/bin/activate
echo "Environment loaded..."

#INPUT_DOCS_DIR=/leonardo_scratch/fast/EUCC_staff_3/lbabetto/vectoria-work/vectoria/test/data/docx # [pdf, docx]
INPUT_DOCS_DIR=/leonardo_scratch/fast/EUCC_staff_3/lbabetto/vectoria-work/docs # new documents
OUTPUT_INDEX_DIR=/leonardo_scratch/fast/EUCC_staff_3/lbabetto/vectoria-work/vectoria/test/index
OUTPUT_SUFFIX=_cin_docs
CONFIG_FILE_PATH=/leonardo_scratch/fast/EUCC_staff_3/lbabetto/vectoria-work/vectoria/etc/default/cineca_config.yaml

echo "START"
python vectoria --config $CONFIG_FILE_PATH build_index --input-docs-dir $INPUT_DOCS_DIR --output-dir $OUTPUT_INDEX_DIR --output-suffix $OUTPUT_SUFFIX
echo "END"
