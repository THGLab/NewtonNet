
from newtonnet.utils.ase_interface import MLAseCalculator
from ase import Atoms

mlcalculator = MLAseCalculator(
    model_path=['training_8/models/best_model_state.tar',
                'training_9/models/best_model_state.tar',
                'training_10/models/best_model_state.tar',
                'training_11/models/best_model_state.tar'],
    settings_path=['training_8/run_scripts/config.yml',
                   'training_9/run_scripts/config.yml',
                   'training_10/run_scripts/config.yml',
                   'training_11/run_scripts/config.yml'],
    method='autograd',    # 'autograd', 'fwd_diff', or None
    #grad_precision=0.0001,    # optional, grid size in Angstrom for Hessian gradient calculation (default: 0.0001)
    device='cpu'   # 'cpu' or list of cuda
)

h2 = Atoms('H2',
           positions=[[0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]])
mlcalculator.calculate(h2)

print(mlcalculator.results)
