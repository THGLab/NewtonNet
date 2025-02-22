import os
import numpy as np
from ase.io import read
from ase.md.langevin import Langevin
from ase.units import fs
from newtonnet.utils.ase_interface import MLAseCalculator


print('Running aspirin MD simulation with NewtonNet...')
atoms = read('md17_data/aspirin/ccsd_test/raw/aspirin_ccsd-test.xyz', index=0)
calc = MLAseCalculator(
    model_path='md17_model/training_1/models/best_model.pt', 
    properties=['energy', 'forces'],
    precision='single',
    device='cuda',
)
atoms.calc = calc
os.makedirs('md17_md')
np.random.seed(0)

dyn = Langevin(
    atoms, 
    timestep=0.5*fs, 
    temperature_K=300, 
    friction=1/(500*fs), 
    logfile='md17_md/md.log', 
    trajectory='md17_md/md.traj', 
    loginterval=100,
    )
dyn.run(20000)
print('MD simulation finished')
