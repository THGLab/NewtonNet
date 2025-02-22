# NewtonNet
A Newtonian message passing network for deep learning of interatomic potentials and forces

## Installation and Dependencies
We recommend using conda environment to install dependencies of this library.
Please install (or load) conda and then proceed with the following commands:

    conda create --name newtonnet python=3.12
    conda activate newtonnet

Now, you can install NewtonNet in the conda environment by cloning this repository:

    git clone https://github.com/THGLab/NewtonNet.git

and then runnig the following command inside the `NewtonNet` repository (where you have access to `pyproject.toml`):

    pip install torch
    pip install -e .

Once you finished installations succesfully, you will be able to run NewtonNet modules anywhere on your computer as long as the `newtonnet` environment is activated. If you have trouble installing `torch_geometric`, `torch_scatter`, or `torch_cluster`, please refer to the [PyG documentation page](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). Optionally, if you want to use [Weights & Biases](https://docs.wandb.ai) for logging, you can initialize it with

    wandb login


## Training and Inference
You can find several run files inside the `scripts` directory that rely on the implemented modules in the NewtonNet library. The run scripts need to be accompanied with a yaml configuration file. You can run an example training script with the following command:

    python newtonnet_train.py --config config.yaml

or resume a checkpoint of an interupted training with the following command:

    python newtonnet_train.py --resume md17_model/training_1

Optionally for large datasets, you might want to process the data on a CPU node with larger memory using:

    python preprocess.py --root md17_data/aspirin/ccsd_train

All models are assumed in [ASE units](https://wiki.fysik.dtu.dk/ase/ase/units.html), such as eV and Ang. You can call an ASE calculator from `newtonnet.utils.ase_interface`. An example MD script can be found in `simulate.py`.

The documentation of the modules are available at most cases. Please look up local classes or functions and consult with the docstrings in the code.

