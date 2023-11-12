# NewtonNet
A Newtonian message passing network for deep learning of interatomic potentials and forces

## Installation and Dependencies
We recommend using conda environment to install dependencies of this library.
Please install (or load) conda and then proceed with the following commands:

    conda create --name newtonnet python=3.10
    conda activate newtonnet

Now, you can install NewtonNet in the conda environment by cloning this repository:

    git clone https://github.com/THGLab/NewtonNet.git

and then runnig the following command inside the NewtonNet repository (where you have access to setup.py):

    pip install -e .

Once you finished installations succesfully, you will be able to run NewtonNet modules
anywhere on your computer as long as the `newtonnet` environment is activated.


## Guidelines
- You can find several run files inside the scripts directory that rely on the implemented modules in the NewtonNet library. 

- The run scripts need to be accompanied with a yaml configuration file.

- The documentation of the modules are available at most cases. Please look up local classes or functions
and consult with the docstrings in the code.

