# EuroCC

The EuroCC project is a part of the European High-Performance Computing Joint Undertaking (EuroHPC JU) and aims to support the development of a European HPC ecosystem. The project is funded by the European Union’s Horizon 2020 research and innovation programme and by the participating countries.

## Installation instructions

```
# CLONE REPOSITORY
git clone git@gitlab.hpc.cineca.it:aproia00/vectoria.git

# LOAD MODULES
module load profile/deeplrn
module load cineca-ai/4.3.0

# CREATE AND ACTIVATE ENV (USING SYS PACKAGES)
python -m venv eucc-env --system-site-packages
source eucc-env/bin/activate

# INSTALL ADDITIONAL REQUIREMENTS
cd vectoria
pip install -e ".[evaluation,nb,test,ui]"
```

# User interface
```
streamlit run vectoria_lib/gui/gui_v1.py
```

# Run tests with
```
pytest -v test/ -m "not slow"
```

# Running vLLM


### Non-quantized model:
```
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --dtype auto  --host 127.0.0.1 --port 8899 --api-key abcd  --gpu-memory-utilization 0.8  --max_model_len 25000
```

### Embedding model:
```
vllm serve BAAI/bge-multilingual-gemma2 --dtype auto  --host 127.0.0.1 --port 8898 --api-key abcd  --gpu-memory-utilization 0.8
```


### Quantized model with AWQ:
```
vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --dtype auto  --host 127.0.0.1 --port 8899 --api-key abcd  --gpu-memory-utilization 0.8  --quantization awq  --max_model_len 25000
```

### Quantized model with GGUF:
https://docs.vllm.ai/en/latest/quantization/gguf.html

Download the model version:
```
wget https://huggingface.co/second-state/E5-Mistral-7B-Instruct-Embedding-GGUF/resolve/main/e5-mistral-7b-instruct-Q5_K_M.gguf
```

Download the tokenizer of the original model
```
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained("intfloat/e5-mistral-7b-instruct")
```
Start vllm:
```
vllm serve ./e5-mistral-7b-instruct-Q5_K_M.gguf --tokenizer intfloat/e5-mistral-7b-instruct --dtype auto --host 127.0.0.1 --port 8899 --api-key abcd --gpu-memory-utilization 0.8 --quantization awq --max_model_len 30000
```

### Big model on HPC node
vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --dtype auto  --host 127.0.0.1 --port 8899 --api-key abcd  --gpu-memory-utilization 0.7  --quantization awq  --max_model_len 25000 --pipeline-parallel-size 4 --cpu--offload-gb 160

