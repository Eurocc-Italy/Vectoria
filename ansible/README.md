# Deploy Vectoria on HPC via Ansible

This Ansible playbook allows you to easily deploy and configure Vectoria on remote HPC. Before continuing, make sure you have Ansible and the necessary prerequisites installed by running the following command from this folder:

```shell
pip install -r requirements.txt
```

>NOTE: it is recommended to run this command in a virtual environment, to avoid installing packages in the base Python install.

## Prerequisites

For the installation, the user launching the command should have ssh access to the remote host. To download some LLMs (such as Llama-3.1), it is also necessary to register your ssh key on the remote host for download. Please follow the install instructions of your LLM of choice.

On HPC, `git-lfs` is used to download LLM files. Make sure this is available on the remote host, either by default or by activating a modulefile.

## Configuration

Modify the `ansible/inventory/hosts` file with the login node (`ansible_host`) and user account (`ansible_user`) to be used for the installation.

If installing locally, edit the `deploy.yml` file changing the variable `hosts` from `hpc` to `localhost`

It is then possible to configure some options for the deploy, such as the paths at which the installation will occur and which LLM models to use or which modulefiles to load. All configurable options are found in the `ansible/roles/vectoria/vars/config.yml` file, with some comments to explain what those options are needed for.

## Deploy

After customising your configuration, run the following command to launch the deploy via Ansible:

```shell
ansible-playbook -i inventory deploy.yml
```
