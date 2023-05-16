
from newtonnet.utils.ase_interface import MLAseCalculator
from ase import Atoms

mlcalculator = MLAseCalculator(model_path='training_18/models/best_model_state.tar',
                               settings_path='training_18/run_scripts/config0.yml',
                               method='fwd_diff',    # 'autograd', 'fwd_diff', or None
                               grad_precision=0.0001,    # optional, grid size in Angstrom for Hessian gradient calculation (default: 0.0001)
                               device='cpu'   # 'cpu' or list of cuda
               )

h2 = Atoms('H2',
           positions=[[0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]])
mlcalculator.calculate(h2)

print(mlcalculator.results)
