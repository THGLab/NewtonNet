# NewtonNet
A Newtonian message passing network for deep learning of interatomic potentials and forces

## Installation and Dependencies
The developer installation is available and for that you need to first clone NewtonNet repository:

    git clone git@github.com:THGLab/NewtonNet.git

and then run the following command inside the NewtonNet repository:

    pip install -e .


We recommend using conda environment to install dependencies of this library.
Please install (or load) conda and then proceed with the following commands:

    conda create --name newtonnet python=3.7
    conda activate torch-gpu
    conda install -c conda-forge numpy scipy scikit-learn pandas ase tqdm
    pip install pyyaml

You also need to install Pytorch based on your hardware (we support both cpu and gpu) and the command line 
provided on the official website: https://pytorch.org/get-started/locally/

Once you finished installations succesfully, you will be able to run NewtonNet modules
anywhere on your computer as long as the `newtonnet` environment is activated.


## Guidelines
- You can find several run files inside the scripts directory that rely on the implemented modules in the NewtonNet library. 

- The run scripts need to be accompanied with a yaml configuration file.

- The documentation of the modules are available at most cases. Please look up local classes or functions
and consult with the docstrings in the code.

