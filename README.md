# EuroCC

The EuroCC project is a part of the European High-Performance Computing Joint Undertaking (EuroHPC JU) and aims to support the development of a European HPC ecosystem. The project is funded by the European Union’s Horizon 2020 research and innovation programme and by the participating countries.

## Installation instructions

```
# CLONE REPOSITORY
git clone git@gitlab.hpc.cineca.it:aproia00/llm_eucc.git

# LOAD MODULES
module load profile/deeplrn
module load cineca-ai/4.3.0

# CREATE AND ACTIVATE ENV (USING SYS PACKAGES)
python -m venv eucc-env --system-site-packages
source eucc-env/bin/activate

# INSTALL ADDITIONAL REQUIREMENTS
cd llm_eucc/eurocc
pip install -e ".[dev,test]"
```

# User interface
```
streamlit run vectoria_lib/gui/gui_v1.py
```


## Configuration
```
Command line
Config file by --config
Config file in /etc
Config file in /etc/default
```
