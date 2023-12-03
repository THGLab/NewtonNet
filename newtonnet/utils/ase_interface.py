import numpy as np
from ase.units import *
from ase.calculators.calculator import Calculator
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.autograd.functional as F

from newtonnet.layers.activations import get_activation_by_string
from newtonnet.layers.cutoff import CosineCutoff, PolynomialCutoff
from newtonnet.layers.scalers import Normalizer
from newtonnet.models import NewtonNet
from newtonnet.data import MolecularDataset


##-------------------------------------
##     ML model ASE interface
##--------------------------------------
class MLAseCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'hessian']

    ### Constructor ###
    def __init__(self, model_path, settings_path, hess_method=None, hess_precision=None, disagreement='std', device='cpu', script=False, trace_n_atoms=False, **kwargs):
        """
        Constructor for MLAseCalculator

        Parameters
        ----------
        model_path: str or list of str
            path to the model. eg. '5k/models/best_model_state.tar'
        settings_path: str or list of str
            path to the .yml setting path. eg. '5k/run_scripts/config_h2.yml'
        hess_method: str
            method to calculate hessians. 
            None: do not calculate hessian (default)
            'autograd': automatic differentiation
            'fwd_diff': forward difference
            'cnt_diff': central difference
        hess_precision: float
            hessian gradient calculation precision for 'fwd_diff' and 'cnt_diff', ignored otherwise (default: None)
        disagreement: str
            method to calculate disagreement between models.
            'std': standard deviation of all votes (default)
            'std_outlierremoval': standard deviation with outlier removal
            'range': range of all votes
            'values': values of each vote
            None: do not calculate disagreement
        device: 
            device to run model. eg. 'cpu', ['cuda:0', 'cuda:1']
        kwargs
        """
        Calculator.__init__(self, **kwargs)

        if type(device) is list:
            self.device = [torch.device(item) for item in device]
        else:
            self.device = [torch.device(device)]
        self.script = script
        self.trace_n_atoms = trace_n_atoms

        self.hess_method = hess_method
        if self.hess_method == 'autograd':
            self.return_hessian = True
        elif self.hess_method == 'fwd_diff' or self.hess_method == 'cnt_diff':
            self.return_hessian = False
            self.hess_precision = hess_precision
        else:
            self.return_hessian = False

        self.disagreement = disagreement

        # torch.set_default_tensor_type(torch.DoubleTensor)
        if type(model_path) is list:
            self.models = [torch.load(model_path_, map_location=self.device[0]) for model_path_ in model_path]
            self.settings = [yaml.safe_load(open(settings_path_, 'r')) for settings_path_ in settings_path]
        else:
            self.models = [torch.load(model_path, map_location=self.device[0])]
            self.settings = [yaml.safe_load(open(settings_path, 'r'))]
        

    def calculate(self, atoms=None, properties=['energy','forces','hessian'], system_changes=None):
        super().calculate(atoms, properties, system_changes)
        data = self.data_formatter(atoms)
        data = MolecularDataset(data)
        gen = DataLoader(dataset=data, batch_size=len(data), shuffle=False, drop_last=False)
        for data in gen:
            data['R'] = data['R'].to(self.device[0])
            data['Z'] = data['Z'].to(self.device[0])
            data['AM'] = data['AM'].to(self.device[0])
            data['N'] = data['N'].to(self.device[0])
            data['NM'] = data['NM'].to(self.device[0])
            data['D'] = data['D'].to(self.device[0])
            data['V'] = data['V'].to(self.device[0])
            energy = np.zeros((len(self.models), 1))
            forces = np.zeros((len(self.models), data['R'].shape[1], 3))
            hessian = np.zeros((len(self.models), data['R'].shape[1], 3, data['R'].shape[1], 3))
            if self.hess_method=='autograd':
                for model_, model in enumerate(self.models):
                    #pred_E = lambda R: self.model(dict(data, R=R))
                    #pred_F = torch.func.jacrev(pred_E)
                    #pred_H = torch.func.hessian(pred_E)
                    #energy = pred_E(data['R'])
                    #forces = -pred_F(data['R'])
                    #hessian = pred_H(data['R'])
                    #energy = self.model(data).detach().cpu().numpy()
                    #forces = -F.jacobian(lambda R: self.model(dict(data, R=R)), data['R']).detach().cpu().numpy()
                    #hessian = F.hessian(lambda R: self.model(dict(data, R=R), vectorize=True), data['R']).detach().cpu().numpy()
                    pred = model(atomic_numbers=data['Z'], positions=data['R'], atom_mask=data['AM'], neighbors=data['N'], neighbor_mask=data['NM'], distances=data['D'], distance_vectors=data['V'])
                    energy[model_] = pred['E'].detach().cpu().numpy() * (kcal/mol)
                    forces[model_] = pred['F'].detach().cpu().numpy() * (kcal/mol/Ang)
                    hessian[model_] = pred['H'].detach().cpu().numpy() * (kcal/mol/Ang/Ang)
                    del pred
            elif self.hess_method=='fwd_diff':
                for model_, model in enumerate(self.models):
                    pred = model(atomic_numbers=data['Z'], positions=data['R'], atom_mask=data['AM'], neighbors=data['N'], neighbor_mask=data['NM'], distances=data['D'], distance_vectors=data['V'])
                    energy[model_] = pred['E'].detach().cpu().numpy()[0] * (kcal/mol)
                    forces_temp = pred['F'].detach().cpu().numpy() * (kcal/mol/Ang)
                    forces[model_] = forces_temp[0]
                    n = 1
                    for A_ in range(data['R'].shape[1]):
                        for X_ in range(3):
                            hessian[model_, A_, X_, :, :] = -(forces_temp[n] - forces_temp[0]) / self.hess_precision
                            n += 1
                    del pred
            elif self.hess_method=='cnt_diff':
                for model_, model in enumerate(self.models):
                    pred = model(atomic_numbers=data['Z'], positions=data['R'], atom_mask=data['AM'], neighbors=data['N'], neighbor_mask=data['NM'], distances=data['D'], distance_vectors=data['V'])
                    energy[model_] = pred['E'].detach().cpu().numpy()[0] * (kcal/mol)
                    forces_temp = pred['F'].detach().cpu().numpy() * (kcal/mol/Ang)
                    forces[model_] = forces_temp[0]
                    n = 1
                    for A_ in range(data['R'].shape[1]):
                        for X_ in range(3):
                            hessian[model_, A_, X_, :, :] = -(forces_temp[n] - forces_temp[n+1]) / 2 / self.hess_precision
                            n += 2
                    del pred
            elif self.hess_method is None:
                for model_, model in enumerate(self.models):
                    pred = model(atomic_numbers=data['Z'], positions=data['R'], atom_mask=data['AM'], neighbors=data['N'], neighbor_mask=data['NM'], distances=data['D'], distance_vectors=data['V'])
                    energy[model_] = pred['E'].detach().cpu().numpy()[0] * (kcal/mol)
                    forces[model_] = pred['F'].detach().cpu().numpy()[0] * (kcal/mol/Ang)
                    del pred
        # energy = np.zeros((len(self.models), 1))
        # forces = np.zeros((len(self.models), len(atoms), 3))
        # hessian = np.zeros((len(self.models), len(atoms), 3, len(atoms), 3))
        # for model_, model in enumerate(self.models):
        #     pred = model(
        #         atomic_numbers=torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device[0])[None, ...], 
        #         positions=torch.tensor(atoms.get_positions(), dtype=torch.float, device=self.device[0])[None, ...], 
        #         atom_mask=torch.ones((1, len(atoms.get_atomic_numbers())), dtype=torch.long, device=self.device[0]), 
        #         neighbors=torch.tensor(atoms.get_all_neighbors(self.settings), dtype=torch.long, device=self.device[0])[None, ...], 
        #         neighbor_mask=torch.ones((1, len(atoms.get_atomic_numbers()), 6), dtype=torch.long, device=self.device[0]), 
        #         distances=torch.tensor(atoms.get_all_distances(), dtype=torch.float, device=self.device[0])[None, ...], 
        #         distance_vectors=torch.tensor(atoms.get_all_displacements(), dtype=torch.float, device=self.device[0])[None, ...],
        #         )
        #     energy[model_] = pred['E'].detach().cpu().numpy()[0] * (kcal/mol)
        #     forces[model_] = pred['F'].detach().cpu().numpy()[0] * (kcal/mol/Ang)

            self.results['outlier'] = self.q_test(energy)
            self.results['energy'] = self.remove_outlier(energy, self.results['outlier']).mean()
            self.results['forces'] = self.remove_outlier(forces, self.results['outlier']).mean(axis=0)
            self.results['hessian'] = self.remove_outlier(hessian, self.results['outlier']).mean(axis=0)
            if self.disagreement=='std':
                self.results['energy_disagreement'] = energy.std()
                self.results['forces_disagreement'] = forces.std(axis=0).max()
                self.results['hessian_disagreement'] = hessian.std(axis=0).max()
            elif self.disagreement=='std_outlierremoval':
                self.results['energy_disagreement'] = self.remove_outlier(energy, self.results['outlier']).std()
                self.results['forces_disagreement'] = self.remove_outlier(forces, self.results['outlier']).std(axis=0).max()
                self.results['hessian_disagreement'] = self.remove_outlier(hessian, self.results['outlier']).std(axis=0).max()
            elif self.disagreement=='range':
                self.results['energy_disagreement'] = (energy.max() - energy.min())
                self.results['forces_disagreement'] = (forces.max(axis=0) - forces.min(axis=0)).max()
                self.results['hessian_disagreement'] = (hessian.max(axis=0) - hessian.min(axis=0)).max()
            elif self.disagreement=='values':
                self.results['energy_disagreement'] = energy
                self.results['forces_disagreement'] = forces
                self.results['hessian_disagreement'] = hessian
            del energy, forces, hessian


    # def load_model(self, model_path, settings_path):
    #     settings = yaml.safe_load(open(settings_path, "r"))
    #     activation = get_activation_by_string(settings['model']['activation'])
    #     if settings['model']['cutoff_network'] == 'poly':
    #         cutoff_network = PolynomialCutoff(cutoff=settings['data']['cutoff'])
    #     elif settings['model']['cutoff_network'] == 'cos':
    #         cutoff_network = CosineCutoff(cutoff=settings['data']['cutoff'])
    #     model = NewtonNet(
    #         n_basis=settings['model']['resolution'],
    #         n_features=settings['model']['n_features'],
    #         activation=activation,
    #         n_layers=settings['model']['n_interactions'],
    #         dropout=settings['training']['dropout'],
    #         max_z=10,
    #         cutoff=settings['data']['cutoff'],  ## data cutoff
    #         cutoff_network=cutoff_network,
    #         train_normalizer=settings['model']['train_normalizer'],
    #         requires_dr=settings['model']['requires_dr'],
    #         device=self.device[0],
    #         create_graph=False,
    #         share_layers=settings['model']['shared_interactions'],
    #         return_hessian=self.return_hessian,
    #         double_update_latent=settings['model']['double_update_latent'],
    #         layer_norm=settings['model']['layer_norm'],
    #         )

    #     model = torch.load(model_path, map_location=self.device[0])
    #     # model.to(self.device[0])
    #     model.eval()

    #     if self.script:
    #         model = torch.jit.script(model)
    #         model.save(model_path[:-4] + '_script.pt')
    #     if self.trace_n_atoms:
    #         model = torch.jit.trace(
    #             model, (
    #                 torch.ones((1, self.trace_n_atoms), dtype=torch.long), 
    #                 torch.rand((1, self.trace_n_atoms, 3), dtype=torch.float), 
    #                 torch.ones((1, self.trace_n_atoms), dtype=torch.long), 
    #                 torch.tile(torch.arange(self.trace_n_atoms), (1, self.trace_n_atoms, 1)),
    #                 1 - torch.eye(self.trace_n_atoms, dtype=torch.long)[None, :, :]))

    #     return model
    

    def data_formatter(self, atoms):
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
        if self.hess_method=='fwd_diff':
            n = data['R'].size
            data['R'] = np.tile(data['R'], (1 + n, 1, 1))
            data['Z'] = np.tile(data['Z'], (1 + n, 1))
            data['E'] = np.tile(data['E'], (1 + n, 1))
            data['F'] = np.tile(data['F'], (1 + n, 1, 1))
            n = 1
            for A_ in range(data['R'].shape[1]):
                for X_ in range(3):
                    data['R'][n, A_, X_] += self.hess_precision
                    n += 1
        if self.hess_method=='cnt_diff':
            n = data['R'].size
            data['R'] = np.tile(data['R'], (1 + 2*n, 1, 1))
            data['Z'] = np.tile(data['Z'], (1 + 2*n, 1))
            data['E'] = np.tile(data['E'], (1 + 2*n, 1))
            data['F'] = np.tile(data['F'], (1 + 2*n, 1, 1))
            n = 1
            for A_ in range(data['R'].shape[1]):
                for X_ in range(3):
                    data['R'][n, A_, X_] += self.hess_precision
                    data['R'][n+1, A_, X_] -= self.hess_precision
                    n += 2
        return data


    # def extensive_data_loader(self, data, device=None):
    #     batch = {'R': data['R'], 'Z': data['Z'], 'E': data['E'], 'F': data['F']}
    #     batch = BatchDataset(batch)
    #     return batch
    

    def remove_outlier(self, data, idx):
        if idx is None:
            return data
        else:
            return np.delete(data, idx, axis=0)


    def q_test(self, data):
        """
        Dixon's Q test for outlier detection

        Parameters
        ----------
        data: 1d array with shape (nlearners,)

        Returns
        -------
        idx: int or None
            the index to be filtered out (return only one index for now as the default is only 4 learners)
        """
        if len(data) < 3:
            idx = None
        else:
            q_ref = { 3: 0.970,  4: 0.829,  5: 0.710, 
                      6: 0.625,  7: 0.568,  8: 0.526,  9: 0.493, 10: 0.466, 
                     11: 0.444, 12: 0.426, 13: 0.410, 14: 0.396, 15: 0.384, 
                     16: 0.374, 17: 0.365, 18: 0.356, 19: 0.349, 20: 0.342, 
                     21: 0.337, 22: 0.331, 23: 0.326, 24: 0.321, 25: 0.317, 
                     26: 0.312, 27: 0.308, 28: 0.305, 29: 0.301, 30: 0.290}.get(len(self.models))  # 95% confidence interval
            sorted_data = np.sort(data, axis=0)
            q_stat_min = (sorted_data[1] - sorted_data[0]) / (sorted_data[-1] - sorted_data[0])
            q_stat_max = (sorted_data[-1] - sorted_data[-2]) / (sorted_data[-1] - sorted_data[0])
            if q_stat_min > q_ref:
                idx = np.argmin(data)
            elif q_stat_max > q_ref:
                idx = np.argmax(data)
            else:
                idx = None
        return idx


# ##-------------------------------------
# ##     ASE interface for Plumed calculator
# ##--------------------------------------
# class PlumedCalculator(Calculator):
#     implemented_properties = ['energy', 'forces']  # , 'stress'
#     def __init__(self, ase_plumed, **kwargs):
#         Calculator.__init__(self, **kwargs)
#         self.ase_plumed = ase_plumed
#         self.counter = 0
#         self.prev_force =None
#         self.prev_energy = None

#     def calculate(self, atoms=None, properties=['forces'],system_changes=None):
#         super().calculate(atoms,properties,system_changes)
#         forces = np.zeros((atoms.get_positions()).shape)
#         energy = 0

#         model_force = np.copy(forces)
#         self.counter += 1
#         # every step() call will call get_forces 2 times, only do plumed once(2nd) to make metadynamics work correctly
#         # there is one call to get_forces when initialize
#         # print(self.counter)
#         # plumed_forces, plumed_energy = self.ase_plumed.external_forces(self.counter , new_forces=forces,
#         #
#         #                                                                delta_forces=True)
#         if self.counter % 2 == 1:
#             plumed_forces,plumed_energy = self.ase_plumed.external_forces((self.counter + 1) // 2 - 1, new_forces=forces,
#                                                             new_energy=energy,delta_forces=True)
#             self.prev_force = plumed_forces
#             self.prev_energy = plumed_energy
#             # print('force diff', np.sum(plumed_forces - model_force))
#         else:
#             plumed_forces = self.prev_force
#             plumed_energy = self.prev_energy
#             # print(self.counter)
#         # if self.counter % 500 == 0:
#         #     print('force diff', np.linalg.norm(plumed_forces - model_force))


#         # delta energy and forces
#         if 'energy' in properties:
#             self.results['energy'] = plumed_energy
#         if 'forces' in properties:
#             self.results['forces'] = plumed_forces

# if __name__ == '__main__':
#     pass
