#!/bin/bash

#SBATCH --account=PhDLR_prod
#SBATCH --job-name=infer
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --time 00:30:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 1 task
#SBATCH --cpus-per-task=32   # 8 cores out of 32
#SBATCH --gres=gpu:4        # 1 gpus per node out of 4
##SBATCH --mem=300000        # memory per node out of 494000MB (481GB)
#SBATCH --exclusive

#SBATCH --qos=boost_qos_dbg

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Avoid fragmentation

export TMPDIR="/leonardo_scratch/large/userinternal/aproia00"
export TMP="/leonardo_scratch/large/userinternal/aproia00"

#-------------------------------------------------------

# TEST: COPY MODEL ON REMOTE NODE'S RAM BEFORE EXECUTION
#MODEL_PATH=/leonardo_scratch/fast/PhDLR_prod/Meta-Llama-3.1-8B-Instruct
MODEL_PATH=/leonardo_scratch/fast/PhDLR_prod/Meta-Llama-3.1-70B-Instruct
RAM_PATH=/dev/shm   

# 2. Copy model to RAM
echo "Copying inference model to RAM..."
start_time=$(date +%s)   # Record the start time
cp -r $MODEL_PATH $RAM_PATH
end_time=$(date +%s)     # Record the end time

# 3. Calculate and display time taken to copy the model
time_taken=$((end_time - start_time))
echo "Model copied to RAM in $time_taken seconds."

#-------------------------------------------------------

# COPY RERANKER MODEL ON REMOTE NODE'S RAM BEFORE EXECUTION
# MODEL_PATH=/leonardo_scratch/fast/PhDLR_prod/bge-reranker-v2-gemma
# RAM_PATH=/dev/shm

# # 2. Copy model to RAM
# echo "Copying reranker model to RAM..."
# start_time=$(date +%s)   # Record the start time
# cp -r $MODEL_PATH $RAM_PATH
# end_time=$(date +%s)     # Record the end time

# # 3. Calculate and display time taken to copy the model
# time_taken=$((end_time - start_time))
# echo "Model copied to RAM in $time_taken seconds."

#-------------------------------------------------------

module load profile/deeplrn
module load cineca-ai/4.3.0
echo "All modules loaded..."

source /leonardo_work/PhDLR_prod/eucc-env/bin/activate
echo "Environment loaded..."

# does not work: we need to pass a directory
#FAISS_INDEX_PATH=/leonardo_work/PhDLR_prod/vectoria/test/index/__leonardo_work__PhDLR_prod__bge-m3_faiss_index_leo_docs.pkl
FAISS_INDEX_PATH=/leonardo_work/PhDLR_prod/vectoria/test/data/index/BAAI__bge-m3_faiss_index # OK
TEST_SET_PATH=/leonardo_work/PhDLR_prod/vectoria/test/data/results/test.json
OUTPUT_DIR=/leonardo_work/PhDLR_prod/vectoria/test/data/results
CONFIG_FILE_PATH=/leonardo_work/PhDLR_prod/vectoria/etc/default/cineca_config.yaml

echo "START"
python vectoria --config $CONFIG_FILE_PATH inference --faiss-index-path $FAISS_INDEX_PATH --test-set-path $TEST_SET_PATH --output-dir $OUTPUT_DIR
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