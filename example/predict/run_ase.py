
from newtonnet.utils.ase_interface import MLAseCalculator
from ase import Atoms

mlcalculator = MLAseCalculator(
    model_path=['training_8/models/best_model_state.tar',
                'training_9/models/best_model_state.tar',
                'training_10/models/best_model_state.tar',
                'training_11/models/best_model_state.tar'],    # path to model file, str or list of str
    settings_path=['training_8/run_scripts/config.yml',
                   'training_9/run_scripts/config.yml',
                   'training_10/run_scripts/config.yml',
                   'training_11/run_scripts/config.yml'],    # path to configuration file, str or list of str
    method='autograd',    # 'autograd', 'fwd_diff', 'cnt_diff', or None (default: 'autograd')
    # grad_precision=1e-5,    # optional, grid size in Angstrom for Hessian gradient calculation (default: None)
    device='cpu'   # 'cpu' or list of cuda
)

h2 = Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
mlcalculator.calculate(h2)

print(mlcalculator.results['energy'])    # mean of calculated molecular energies, shape (1,)
print(mlcalculator.results['forces'])    # mean of calculated atomic forces, shape (n_atom, 3)
print(mlcalculator.results['hessian'])    # mean of calculated atomic Hessian, shape (n_atom, 3, n_atom, 3)
print(mlcalculator.results['energy_std'])    # standard deviation of calculated molecular energies, shape (1,)
print(mlcalculator.results['forces_std'])    # standard deviation of calculated atomic forces, shape (n_atom, 3)
print(mlcalculator.results['hessian_std'])    # standard deviation of calculated atomic Hessian, shape (n_atom, 3, n_atom, 3)
print(mlcalculator.results['outlier'])    # index of model that predicts an energy outlier, int