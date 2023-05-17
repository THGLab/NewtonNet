import numpy as np
from ase.units import *
from ase.calculators.calculator import Calculator
import torch
import torch.autograd.functional as F
import yaml

from newtonnet.layers.activations import get_activation_by_string
from newtonnet.models import NewtonNet
from newtonnet.data import ExtensiveEnvironment
from newtonnet.data import batch_dataset_converter


##-------------------------------------
##     ML model ASE interface
##--------------------------------------
class MLAseCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'hessian']

    ### Constructor ###
    def __init__(self, model_path, settings_path, method='fwd_diff', grad_precision=0.0001, device='cpu', **kwargs):
        """
        Constructor for MLAseCalculator

        Parameters
        ----------
        model_path: str
            path to the model. eg. '5k/models/best_model_state.tar'
        settings_path: str
            path to the .yml setting path. eg. '5k/run_scripts/config_h2.yml'
        method: str
            method to calculate hessians. 
            'autograd': automatic differentiation
            'fwd_diff': forward difference (default, requires grad_precision)
            None: do not calculated hessian
        grad_precision: float
            hessian gradient calculation precision. default: 0.0001
        device: 
            device to run model. eg. 'cpu', ['cuda:0', 'cuda:1']
        kwargs
        """
        Calculator.__init__(self, **kwargs)
        self.settings = yaml.safe_load(open(settings_path, "r"))

        if type(device) is list:
            self.device = [torch.device(item) for item in device]
        else:
            self.device = [torch.device(device)]

        self.method = method
        if self.method == 'autograd':
            self.return_hessian = True
        elif self.method == 'fwd_diff':
            self.return_hessian = True
            self.grad_precision = grad_precision
        else:
            self.return_hessian = False

        torch.set_default_tensor_type(torch.DoubleTensor)
        self._load_model(model_path)
        

    def calculate(self, atoms=None, properties=['energy','forces','hessian'],system_changes=None):
        super().calculate(atoms,properties,system_changes)
        data = extensive_data_loader(data=self.data_formatter(atoms),
                                     env_provider=ExtensiveEnvironment(),
                                     batch_size=1, 
                                     n_rotations=0,
                                     device=self.device[0])
        if self.method=='autograd':
            #pred_E = lambda R: self.model(dict(data, R=R))
            #pred_F = torch.func.jacrev(pred_E)
            #pred_H = torch.func.hessian(pred_E)
            #energy = pred_E(data['R'])
            #forces = -pred_F(data['R'])
            #hessian = pred_H(data['R'])
            #energy = self.model(data).detach().cpu().numpy()
            #forces = -F.jacobian(lambda R: self.model(dict(data, R=R)), data['R']).detach().cpu().numpy()
            #hessian = F.hessian(lambda R: self.model(dict(data, R=R), vectorize=True), data['R']).detach().cpu().numpy()
            pred = self.model(data)
            energy = pred['E'].detach().cpu().numpy()
            forces = pred['F'].detach().cpu().numpy()
            hessian = pred['H'].detach().cpu().numpy()
        elif self.method=='fwd_diff':
            pred = self.model(data)
            energy = pred['E'].detach().cpu().numpy()
            forces = pred['F'].detach().cpu().numpy()
            hessian = np.zeros((1, forces.shape[1], 3, forces.shape[1], 3))
            n = 1
            for A_ in range(forces.shape[1]):
                for X_ in range(3):
                    hessian[0, A_, X_, :, :] = -(forces[n] - forces[0]) / self.grad_precision
                    n += 1
            forces = forces[0]
        energy = energy * kcal / mol
        forces = forces * kcal / mol / Ang
        hessian =  hessian * kcal / mol / Ang / Ang
        if 'energy' in properties:
            self.results['energy'] = energy.squeeze()
        if 'forces' in properties:
            self.results['forces'] = forces.squeeze()
        if 'hessian' in properties:
            self.results['hessian'] = hessian.squeeze()
        del pred, energy, forces, hessian


    def _load_model(self,model_path):
        settings = self.settings
        activation = get_activation_by_string(settings['model']['activation'])
        model = NewtonNet(resolution=settings['model']['resolution'],
                            n_features=settings['model']['n_features'],
                            activation=activation,
                            n_interactions=settings['model']['n_interactions'],
                            dropout=settings['training']['dropout'],
                            max_z=10,
                            cutoff=settings['data']['cutoff'],  ## data cutoff
                            cutoff_network=settings['model']['cutoff_network'],
                            normalize_atomic=settings['model']['normalize_atomic'],
                            requires_dr=settings['model']['requires_dr'],
                            device=self.device[0],
                            create_graph=False,
                            shared_interactions=settings['model']['shared_interactions'],
                            return_hessian=self.return_hessian,
                            double_update_latent=settings['model']['double_update_latent'],
                            layer_norm=settings['model']['layer_norm'],
                            )

        model.load_state_dict(torch.load(model_path, map_location=self.device[0])['model_state_dict'], )
        self.model = model
        self.model.to(self.device[0])
        self.model.eval()
        self.model.requires_dr = True


    def predict(self,data_dict):
        env = ExtensiveEnvironment()
        val_batch = extensive_data_loader(data=data_dict,
                                         env_provider=env,
                                         batch_size=1, 
                                         n_rotations=0,
                                         device=self.device[0],)
        #val_batch = next(data_gen)
        data_preds = self.model(val_batch)

        return data_preds


    def data_formatter(self,atoms):
        """
        convert ase.Atoms to input format of the model

        Parameters
        ----------
        atoms: ase.Atoms

        Returns
        -------
        data: dict
            dictionary of arrays with following keys:
                - 'R':positions
                - 'Z':atomic_numbers
                - 'E':energy
                - 'F':forces
        """
        data  = {
            'R': np.array(atoms.get_positions())[np.newaxis, ...], #shape(ndata,natoms,3)
            'Z': np.array(atoms.get_atomic_numbers())[np.newaxis, ...], #shape(ndata,natoms)
            'E': np.zeros((1,1)), #shape(ndata,1)
            'F': np.zeros((1,len(atoms.get_atomic_numbers()), 3)),#shape(ndata,natoms,3)
        }
        if self.method=='fwd_diff':
            n = data['R'].size
            data['R'] = np.tile(data['R'], (1 + n, 1, 1))
            data['Z'] = np.tile(data['Z'], (1 + n, 1))
            data['E'] = np.tile(data['E'], (1 + n, 1))
            data['F'] = np.tile(data['F'], (1 + n, 1, 1))
            n = 1
            for A_ in range(data['R'].shape[1]):
                for X_ in range(3):
                    data['R'][n, A_, X_] += self.grad_precision
                    n += 1
        return data


def extensive_data_loader(data,
                          env_provider=None,
                          batch_size=32,
                          n_rotations=0,
                          device=None,
                         ):
    r"""
    The main function to load and iterate data based on the extensive environment provider.

    Parameters
    ----------
    data: dict
        dictionary of arrays with following keys:
            - 'R':positions
            - 'Z':atomic_numbers
            - 'E':energy
            - 'F':forces
            optional:
            - 'lattice': lattice vector for pbc shape(9,)

    env_provider: ShellProvider
        the instance of combust.data.ExtensiveEnvironment

    batch_size: int, optional (default: 32)
        The size of output tensors

    n_rotations: int, optional (default: 0)
        Number of times to rotate voxel boxes for data augmentation.
        If zero, the original orientation will be used.

    freeze_rotations: bool, optional (default: False)
        If True rotation angles will be determined and fixed during generation.

    keep_original: bool, optional (default: True)
        If True, the original orientation of data is kept in each epoch

    device: torch.device
        either cpu or gpu (cuda) device.

    shuffle: bool, optional (default: True)
        If ``True``, shuffle the list of file path and batch indices between iterations.

    drop_last: bool, optional (default: False)
        set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (default: ``False``)

    Yields
    -------
    BatchDataset: instance of BatchDataset with the all batch data

    """
    batch = {'R': torch.tensor(data['R']),
             'Z': torch.tensor(data['Z'])}
    N, NM, AM, _, _ = ExtensiveEnvironment().get_environment(batch['R'].clone(), batch['Z'].clone())
    batch.update({'N': N, 'NM': NM, 'AM': AM})
    batch = batch_dataset_converter(batch, device=device)
    return batch


##-------------------------------------
##     ASE interface for Plumed calculator
##--------------------------------------
class PlumedCalculator(Calculator):
    implemented_properties = ['energy', 'forces']  # , 'stress'
    def __init__(self, ase_plumed, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.ase_plumed = ase_plumed
        self.counter = 0
        self.prev_force =None
        self.prev_energy = None

    def calculate(self, atoms=None, properties=['forces'],system_changes=None):
        super().calculate(atoms,properties,system_changes)
        forces = np.zeros((atoms.get_positions()).shape)
        energy = 0

        model_force = np.copy(forces)
        self.counter += 1
        # every step() call will call get_forces 2 times, only do plumed once(2nd) to make metadynamics work correctly
        # there is one call to get_forces when initialize
        # print(self.counter)
        # plumed_forces, plumed_energy = self.ase_plumed.external_forces(self.counter , new_forces=forces,
        #
        #                                                                delta_forces=True)
        if self.counter % 2 == 1:
            plumed_forces,plumed_energy = self.ase_plumed.external_forces((self.counter + 1) // 2 - 1, new_forces=forces,
                                                            new_energy=energy,delta_forces=True)
            self.prev_force = plumed_forces
            self.prev_energy = plumed_energy
            # print('force diff', np.sum(plumed_forces - model_force))
        else:
            plumed_forces = self.prev_force
            plumed_energy = self.prev_energy
            # print(self.counter)
        # if self.counter % 500 == 0:
        #     print('force diff', np.linalg.norm(plumed_forces - model_force))


        # delta energy and forces
        if 'energy' in properties:
            self.results['energy'] = plumed_energy
        if 'forces' in properties:
            self.results['forces'] = plumed_forces

if __name__ == '__main__':
    pass
