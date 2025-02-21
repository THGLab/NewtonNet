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


##-------------------------------------
##     ML model ASE interface
##--------------------------------------
class MLAseCalculator(Calculator):
    implemented_properties = ['energy', 'free_energy', 'forces', 'hessian', 'stress']
    # note that the free_energy is not the Gibbs/Helmholtz free energy, but the potential energy in the ASE calculator, how confusing

    ### Constructor ###
    def __init__(
            self, 
            model_path: str,
            properties: list = ['energy', 'free_energy', 'forces'], 
            disagreement: str = None,
            device: str | list[str] = 'cpu',
            precision: str = 'float32',
            script: bool = False,
            trace_n_atoms: int = 0,
            **kwargs,
            ):
        """
        Constructor for MLAseCalculator for NewtonNet models

        Parameters:
            model_path (str): The path to the model.
            settings_path (str): The path to the settings.
            properties (list): The properties to be predicted. Default: ['energy', 'free_energy', 'forces'].
            disagreement (str): The type of disagreement to be calculated. Default: None.
            device (str): The device for the calculator. Default: 'cpu'.
            precision (str): The precision of the calculator. Default: 'float32'.
            script (bool): Whether to script the model. Default: False.
            trace_n_atoms (int): The number of atoms to be traced. Default: 0.
        """
        Calculator.__init__(self, **kwargs)

        if type(device) is list:
            self.device = [torch.device(item) for item in device]
        else:
            self.device = [torch.device(device)]
        self.dtype = get_precision_by_string(precision)

        self.models = []
        self.properties = properties
        if isinstance(model_path, list):
            self.models = [self.load_model(model) for model in model_path]
        else:
            self.models = [self.load_model(model_path)]
        
        self.radius_graph = RadiusGraph(self.models[0].embedding_layer.norm.r)
        
        self.script = script
        self.trace_n_atoms = trace_n_atoms
        self.disagreement = disagreement
        

    def calculate(self, atoms=None, properties=['energy', 'forces', 'stress'], system_changes=None):
        super().calculate(atoms, self.properties, system_changes)
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        data = self.format_data(atoms)
        n_models, n_frames, n_atoms = len(self.models), len(atoms), len(atoms[0])

        preds = {}
        if 'energy' in self.properties:
            preds['energy'] = np.zeros((n_models, n_frames))
        if 'free_energy' in self.properties:
            preds['free_energy'] = np.zeros((n_models, n_frames))
        if 'forces' in self.properties:
            preds['forces'] = np.zeros((n_models, n_frames, n_atoms, 3))
        if 'hessian' in self.properties:
            preds['hessian'] = np.zeros((n_models, n_frames, n_atoms, 3, n_atoms, 3))
        if 'stress' in self.properties:
            preds['stress'] = np.zeros((n_models, n_frames, 6))
        for model_, model in enumerate(self.models):
            pred = model(data.z, data.disp, data.edge_index, data.batch)
            if 'energy' in self.properties:
                energy = pred.energy.cpu().detach().numpy()
                preds['energy'][model_] = energy
            if 'free_energy' in self.properties:
                energy = pred.energy.cpu().detach().numpy()
                preds['free_energy'][model_] = energy
            if 'forces' in self.properties:
                force = pred.gradient_force.cpu().detach().numpy()
                preds['forces'][model_] = force.reshape(n_frames, n_atoms, 3)
            if 'hessian' in self.properties:
                hessian = pred.hessian.cpu().detach().numpy()
                preds['hessian'][model_] = hessian.reshape(n_frames, n_atoms, 3, n_atoms, 3)
            if 'stress' in self.properties:
                virial = pred.virial.cpu().detach().numpy()
                volume = np.array([atoms[frame].get_volume() for frame in range(n_frames)])
                stress = -virial[:, [0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]] / volume[:, None] / 2
                preds['stress'][model_] = stress
            del pred

        # self.results['outlier'] = self.q_test(preds['energy'])
        for key in self.properties:
            # self.results[key] = self.remove_outlier(preds[key], self.results['outlier']).mean(axis=0)
            self.results[key] = preds[key].mean(axis=0).squeeze()

            # if self.disagreement == 'std':
            #     self.results[key + '_disagreement'] = preds[key].std(axis=0).max()
            # elif self.disagreement == 'std_outlierremoval':
            #     self.results[key + '_disagreement'] = self.remove_outlier(preds[key], self.results['outlier']).std(axis=0).max()
            # elif self.disagreement == 'range':
            #     self.results[key + '_disagreement'] = (preds[key].max(axis=0) - preds[key].min(axis=0)).max()
            # elif self.disagreement == 'values':
            #     self.results[key + '_disagreement'] = preds[key]
            del preds[key]

    def load_model(self, model):
        # TODO: Load model with only weights
        model = torch.load(model, map_location=self.device[0])
        keys_to_keep = ['energy']
        for key in self.properties:
            key = {
                'energy': 'energy',
                'free_energy': 'energy',
                'forces': 'gradient_force',
                'stress': 'virial',
                'hessian': 'hessian',
            }.get(key)
            keys_to_keep.append(key)
            if key in model.infer_properties:
                continue
            model.infer_properties.append(key)
            model.output_layers.append(get_output_by_string(key))
            model.scalers.append(get_scaler_by_string(key))
            model.aggregators.append(get_aggregator_by_string(key))
        ids_to_remove = [i for i, key in enumerate(model.infer_properties) if key not in keys_to_keep]
        for i in reversed(ids_to_remove):
            model.infer_properties.pop(i)
            model.output_layers.pop(i)
            model.scalers.pop(i)
            model.aggregators.pop(i)
        model.to(self.dtype)
        model.eval()
        model.embedding_layer.requires_dr = any(isinstance(layer, DerivativeProperty) for layer in model.output_layers)
        if any(isinstance(layer, SecondDerivativeProperty) for layer in model.output_layers):
            for layer in model.output_layers:
                if isinstance(layer, DerivativeProperty):
                    layer.create_graph = True
        return model

    def format_data(self, atoms_list):
        data_list = []
        for atoms in atoms_list:
            z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device[0])
            pos = torch.tensor(atoms.get_positions(wrap=True), dtype=self.dtype, device=self.device[0])
            lattice = torch.tensor(atoms.get_cell().array, dtype=self.dtype, device=self.device[0])
            lattice[~atoms.get_pbc()] = torch.inf
            data = Data(pos=pos, z=z, lattice=lattice)
            data = self.radius_graph(data)
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
    