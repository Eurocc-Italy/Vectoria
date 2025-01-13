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

#-------------------------------------------------------

# TEST: COPY MODEL ON REMOTE NODE'S RAM BEFORE EXECUTION
MODEL_PATH=/path/to/installdir/inference_engine
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
# NOTE: comment the following block if reranker is not used
MODEL_PATH=/path/to/installdir/reranker_model
RAM_PATH=/dev/shm

# 2. Copy model to RAM
echo "Copying reranker model to RAM..."
start_time=$(date +%s)   # Record the start time
cp -r $MODEL_PATH $RAM_PATH
end_time=$(date +%s)     # Record the end time

# 3. Calculate and display time taken to copy the model
time_taken=$((end_time - start_time))
echo "Model copied to RAM in $time_taken seconds."

#-------------------------------------------------------

# Load necessary modules
module load <...>
echo "All modules loaded..."

source /path/to/installdir/vectoria_env/bin/activate
echo "Environment loaded..."

# does not work: we need to pass a directory
INDEX_PATH=/path/to/installdir/vectoria/test/data/index/<INDEX_NAME>
TEST_SET_PATH=/path/to/installdir/vectoria/test/data/results/test.json
OUTPUT_DIR=/path/to/installdir/vectoria/test/data/results
CONFIG_FILE_PATH=/path/to/installdir/vectoria/etc/default/default_config.yaml

echo "START"
# OPTION 1: QUESTIONS FROM FILE
# python vectoria --config $CONFIG_FILE_PATH inference --index-path $INDEX_PATH --test-set-path $TEST_SET_PATH --output-dir $OUTPUT_DIR
# OPTION 2: QUESTIONS FROM CLI
python vectoria --config $CONFIG_FILE_PATH inference --index-path $INDEX_PATH --questions "<YOUR QUESTION GOES HERE>" --output-dir $OUTPUT_DIR
echo "END"
