import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

import torch
from torch_geometric.data import Data, Batch

from newtonnet.layers.precision import get_precision_by_string
from newtonnet.layers.scalers import get_scaler_by_string
from newtonnet.models.output import get_output_by_string, get_aggregator_by_string
from newtonnet.models.output import DerivativeProperty, SecondDerivativeProperty
from newtonnet.data import RadiusGraph
from newtonnet.utils.pretrained_models import download_checkpoint

##-------------------------------------
##     ML model ASE interface
##--------------------------------------
class MLAseCalculator(Calculator):
    implemented_properties = ['charges', 'bec', 'energy', 'free_energy', 'forces', 'hessian', 'stress']
    # note that the free_energy is not the Gibbs/Helmholtz free energy, but the potential energy in the ASE calculator, how confusing

    ### Constructor ###
    def __init__(
            self, 
            model_path: str,
            properties: list = None, 
            device: str = None,
            precision: str = 'float32',
            **kwargs,
            ):
        """
        Constructor for MLAseCalculator for NewtonNet models

        Parameters:
            model_path (str): The path to the model.
            properties (list): The properties to be predicted. Default: model.output_properties.
            device (str): The device for the calculator.
            precision (str): The precision of the calculator. Default: 'float32'.
        """
        Calculator.__init__(self, **kwargs)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.dtype = get_precision_by_string(precision)

        self.properties = properties
        self.model = self.load_model(model_path)
        

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, self.properties, system_changes)
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        data = self.format_data(atoms)
        n_frames, n_atoms = len(atoms), len(atoms[0])

        pred = self.model(data.z, data.pos, data.cell, data.batch)
        if 'charges' in self.properties:
            charge = pred.charge.cpu().detach().numpy()
            self.results['charges'] = charge.reshape(n_frames, n_atoms).squeeze()
        if 'bec' in self.properties:
            bec = pred.bec.cpu().detach().numpy()
            self.results['bec'] = bec.reshape(n_frames, n_atoms, 3, 3).squeeze()
        if 'energy' in self.properties:
            energy = pred.energy.cpu().detach().numpy()
            self.results['energy'] = energy.squeeze()
        if 'free_energy' in self.properties:
            energy = pred.energy.cpu().detach().numpy()
            self.results['free_energy'] = energy.squeeze()
        if 'forces' in self.properties:
            force = pred.gradient_force.cpu().detach().numpy()
            self.results['forces'] = force.reshape(n_frames, n_atoms, 3).squeeze()
        if 'hessian' in self.properties:
            hessian = pred.hessian.cpu().detach().numpy()
            self.results['hessian'] = hessian.reshape(n_frames, n_atoms, 3, n_atoms, 3).squeeze()
        if 'stress' in self.properties:
            stress = pred.stress.cpu().detach().numpy()
            self.results['stress'] = stress[:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]].squeeze()
        del pred

    def load_model(self, model):
        # TODO: Load model with only weights
        if model in ['ani1', 'ani1x', 't1x']:
            model = download_checkpoint(model)
        model = torch.load(model, map_location=self.device, weights_only=False)
        if self.properties is None:
            self.properties = []
            for key in model.output_properties:
                key = {
                    'charge': 'charges',
                    'energy': 'energy',
                    'gradient_force': 'forces',
                }.get(key)
                self.properties.append(key)
        else:
            keys_to_keep = ['charge', 'energy']
            for key in self.properties:
                key = {
                    'charges': 'charge',
                    'bec': 'bec',
                    'energy': 'energy',
                    'free_energy': 'energy',
                    'forces': 'gradient_force',
                    'stress': 'stress',
                    'hessian': 'hessian',
                }.get(key)
                keys_to_keep.append(key)
                if key in model.output_properties:
                    continue
                model.output_properties.append(key)
                model.output_layers.append(get_output_by_string(key))
                model.scalers.append(get_scaler_by_string(key))
                model.aggregators.append(get_aggregator_by_string(key))
            ids_to_remove = [i for i, key in enumerate(model.output_properties) if key not in keys_to_keep]
            for i in reversed(ids_to_remove):
                model.output_properties.pop(i)
                model.output_layers.pop(i)
                model.scalers.pop(i)
                model.aggregators.pop(i)
        model.to(self.dtype)
        model.eval()
        model.embedding_layers.requires_dr = any(isinstance(layer, DerivativeProperty) for layer in model.output_layers)
        if any(isinstance(layer, SecondDerivativeProperty) for layer in model.output_layers):
            for layer in model.output_layers:
                if isinstance(layer, DerivativeProperty):
                    layer.create_graph = True
        return model

    def format_data(self, atoms_list):
        data_list = []
        for atoms in atoms_list:
            z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
            pos = torch.tensor(atoms.get_positions(wrap=True), dtype=self.dtype, device=self.device)
            cell = torch.tensor(atoms.get_cell().array, dtype=self.dtype, device=self.device)
            pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool, device=self.device)
            cell[~pbc] = 0.0
            data = Data(pos=pos, z=z, cell=cell.reshape(1, 3, 3))
            data_list.append(data)
        batch = Batch.from_data_list(data_list)
        return batch

    # def remove_outlier(self, data, idx):
    #     if idx is None:
    #         return data
    #     else:
    #         return np.delete(data, idx, axis=0)

    # def q_test(self, data):
    #     '''
    #     Dixon's Q test for outlier detection

    #     Parameters:
    #         data (1d array): The data to be tested.

    #     Returns:
    #         idx (int): The index to be filtered out.
    #     '''
    #     idx = np.full_like(data, np.nan)
    #     if len(data) >= 3:
    #         q_ref = { 3: 0.970,  4: 0.829,  5: 0.710, 
    #                   6: 0.625,  7: 0.568,  8: 0.526,  9: 0.493, 10: 0.466, 
    #                  11: 0.444, 12: 0.426, 13: 0.410, 14: 0.396, 15: 0.384, 
    #                  16: 0.374, 17: 0.365, 18: 0.356, 19: 0.349, 20: 0.342, 
    #                  21: 0.337, 22: 0.331, 23: 0.326, 24: 0.321, 25: 0.317, 
    #                  26: 0.312, 27: 0.308, 28: 0.305, 29: 0.301, 30: 0.290}.get(len(self.models))  # 95% confidence interval
    #         sorted_data = np.sort(data, axis=0)
    #         q_stat_min = (sorted_data[1] - sorted_data[0]) / (sorted_data[-1] - sorted_data[0])
    #         q_stat_max = (sorted_data[-1] - sorted_data[-2]) / (sorted_data[-1] - sorted_data[0])
    #         min_mask = q_stat_min > q_ref
    #         max_mask = q_stat_max > q_ref
    #         idx[min_mask] = np.argmin(data[:, min_mask], axis=0)
    #         idx[max_mask] = np.argmax(data[:, max_mask], axis=0)
    #     return idx
