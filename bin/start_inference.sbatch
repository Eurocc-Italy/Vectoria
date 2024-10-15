#!/bin/bash

#SBATCH --account=PhDLR_prod
#SBATCH --job-name=infer
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --time 00:30:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 1 task
##SBATCH --cpus-per-task=8   # 8 cores out of 32
#SBATCH --gres=gpu:1        # 1 gpus per node out of 4
##SBATCH --mem=300000        # memory per node out of 494000MB (481GB)
#SBATCH --exclusive

##SBATCH --qos=boost_qos_dbg

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Avoid fragmentation

export TMPDIR="/leonardo_scratch/large/userinternal/aproia00"
export TMP="/leonardo_scratch/large/userinternal/aproia00"

module load profile/deeplrn
module load cineca-ai/4.3.0
echo "All modules loaded..."

source /leonardo_work/PhDLR_prod/eucc-env/bin/activate
echo "Environment loaded..."

FAISS_INDEX_PATH=/leonardo_work/PhDLR_prod/vectoria/test/index/__leonardo_work__PhDLR_prod__bge-m3_faiss_index_leo_docs.pkl
K=5
TEST_SET_PATH=/leonardo_work/PhDLR_prod/vectoria/test/data/results/led_question_answer.json
OUTPUT_DIR=/leonardo_work/PhDLR_prod/vectoria/test/data/results

echo "START"
python vectoria inference --faiss-index-path $FAISS_INDEX_PATH --k $K --test-set-path $TEST_SET_PATH --output-dir $OUTPUT_DIR
echo "END"

# Interactive node
#salloc --account=PhDLR_prod \
#       --job-name=infer \
#       --partition=boost_usr_prod \
#       --time=00:15:00 \
#       -N 1 \
#       --ntasks-per-node=1 \
#       --cpus-per-task=8 \
#       --gres=gpu:1 \
#       --mem=300000 \
#       --exclusive

# srun ...