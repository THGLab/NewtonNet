# NewtonNet
A Newtonian message passing network for deep learning of interatomic potentials and forces

## Installation and Dependencies
We recommend using conda environment to install dependencies of this library first.
Please install (or load) conda and then proceed with the following commands:

    conda create --name newtonnet python=3.7
    conda activate newtonnet
    conda install -c conda-forge numpy scipy scikit-learn pandas ase tqdm
    pip install pyyaml

You also need to install Pytorch based on your hardware (we support both cpu and gpu) and the command line 
provided on the official website: https://pytorch.org/get-started/locally/. For example:

    conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

Now, you can install NewtonNet in the conda environment by cloning this repository:

    git clone https://github.com/ericyuan00000/NewtonNet.git

and then runnig the following command inside the NewtonNet repository (where you have access to setup.py):

    pip install -e .

Once you finished installations succesfully, you will be able to run NewtonNet modules
anywhere on your computer as long as the `newtonnet` environment is activated.


## Guidelines
- You can find several run files inside the scripts directory that rely on the implemented modules in the NewtonNet library. 

- The run scripts need to be accompanied with a yaml configuration file.

- The documentation of the modules are available at most cases. Please look up local classes or functions
and consult with the docstrings in the code.

