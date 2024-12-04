import os
import os.path as osp
from tqdm import tqdm
from typing import Callable, List, Optional, Union
import numpy as np
import ase
from ase import units
from ase.io import read
setattr(units, 'kcal/mol', units.kcal / units.mol)
setattr(units, 'kJ/mol', units.kJ / units.mol)

import torch
import torch.nn as nn
from torch_geometric.data import Dataset, InMemoryDataset, Data
from torch_geometric.utils import scatter


class MolecularDataset(Dataset):
    '''
    This class is a dataset for molecular data.
    
    Args:
        root (str): The root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in a data object and returns a transformed version. The data object will be transformed before every access. Default: None.
        pre_transform (callable, optional): A function/transform that takes in a data object and returns a transformed version. The data object will be transformed before being saved to disk. Default: None.
        pre_filter (callable, optional): A function that takes in a data object and returns a boolean value, indicating whether the data object should be included in the final dataset. Default: None.
        force_reload (bool): Whether to re-process the dataset. Default: False.
        precision (torch.dtype): The precision of the data. Default: torch.float.
    '''
    def __init__(
        self,
        precision: torch.dtype = torch.float,
        length_unit: str = 'Ang',
        energy_unit: str = 'eV',
        **kwargs,
    ) -> None:
        self.precision = precision
        self.units = {'length': getattr(units, length_unit), 'energy': getattr(units, energy_unit)}
        super().__init__(**kwargs)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str]]:
        names = [name for name in os.listdir(self.raw_dir) if name.endswith(('.npz', '.xyz', '.extxyz'))]
        return names

    # @property
    def processed_file_names(self) -> List[str]:
        if not osp.exists(self.processed_dir):
            return []
        return [name for name in os.listdir(self.processed_dir) if name.startswith('data_') and name.endswith('.pt')]

    def process(self) -> None:
        idx = 0
        for raw_path in tqdm(self.raw_paths):
            if raw_path.endswith('.npz'):
                data_list = parse_npz(raw_path, self.pre_transform, self.pre_filter, self.precision, self.units)
            elif raw_path.endswith('.xyz') or raw_path.endswith('.extxyz'):
                data_list = parse_xyz(raw_path, self.pre_transform, self.pre_filter, self.precision, self.units)
            
            for data in data_list:
                torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1

    def len(self) -> int:
        return len(self.processed_file_names())
    
    def get(self, idx: int) -> Data:
        return torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
    
class MolecularInMemoryDataset(InMemoryDataset):
    '''
    This class is an in-memeory dataset for molecular data.
    
    Args:
        root (str): The root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in a data object and returns a transformed version. The data object will be transformed before every access. Default: None.
        pre_transform (callable, optional): A function/transform that takes in a data object and returns a transformed version. The data object will be transformed before being saved to disk. Default: None.
        pre_filter (callable, optional): A function that takes in a data object and returns a boolean value, indicating whether the data object should be included in the final dataset. Default: None.
        force_reload (bool): Whether to re-process the dataset. Default: False.
        precision (torch.dtype): The precision of the data. Default: torch.float.
    '''
    def __init__(
        self,
        precision: torch.dtype = torch.float,
        length_unit: str = 'Ang',
        energy_unit: str = 'eV',
        **kwargs,
    ) -> None:
        self.precision = precision
        self.units = {'length': getattr(units, length_unit), 'energy': getattr(units, energy_unit)}
        super().__init__(**kwargs)

        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str]]:
        names = [name for name in os.listdir(self.raw_dir) if name.endswith(('.npz', '.xyz', '.extxyz'))]
        return names

    # @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def process(self) -> None:
        data_list = []
        data_path = self.processed_paths[0]
        for raw_path in tqdm(self.raw_paths):
            if raw_path.endswith('.npz'):
                data_list.extend(parse_npz(raw_path, self.pre_transform, self.pre_filter, self.precision, self.units))
            elif raw_path.endswith('.xyz') or raw_path.endswith('.extxyz'):
                data_list.extend(parse_xyz(raw_path, self.pre_transform, self.pre_filter, self.precision, self.units))
            
        self.save(data_list, data_path)

def parse_npz(raw_path: str, pre_transform: Callable, pre_filter: Callable, precision: torch.dtype, units: dict) -> List[Data]:
    data_list = []
    raw_data = np.load(raw_path)

    z = torch.from_numpy(raw_data['Z']).int()
    pos = torch.from_numpy(raw_data['R']).to(precision)
    lattice = torch.from_numpy(raw_data['L']).to(precision) if 'L' in raw_data else torch.ones(3, dtype=precision) * torch.inf
    if lattice.numel() == 3:
        lattice = lattice.diag()
    elif lattice.numel() == 9:
        lattice = lattice.reshape(3, 3)
    else:
        raise ValueError('The lattice must be a single 3x3 matrix for each npz file.')
    energy = torch.from_numpy(raw_data['E']).to(precision) if 'E' in raw_data else None
    force = torch.from_numpy(raw_data['F']).to(precision) if 'F' in raw_data else None

    for i in range(pos.size(0)):
        data = Data()
        data.z = z.reshape(-1) if z.dim() < 2 else z[i].reshape(-1)
        data.pos = pos[i].reshape(-1, 3) * units['length']
        data.lattice = lattice.reshape(1, 3, 3) * units['length']
        if energy is not None:
            data.energy = energy[i].reshape(1) * units['energy']
        if force is not None:
            data.force = force[i].reshape(-1, 3) * units['energy'] / units['length']

        if pre_filter is not None and not pre_filter(data):
            continue
        if pre_transform is not None:
            data = pre_transform(data)
        data_list.append(data)

    return data_list

def parse_xyz(raw_path: str, pre_transform: Callable, pre_filter: Callable, precision: torch.dtype, units: dict) -> List[Data]:
    data_list = []
    atoms_list = ase.io.read(raw_path, index=':')

    for atoms in atoms_list:
        atoms.set_constraint()
        z = torch.from_numpy(atoms.get_atomic_numbers()).int()
        pos = torch.from_numpy(atoms.get_positions(wrap=True)).to(precision)
        lattice = torch.from_numpy(atoms.get_cell().array).to(precision)
        lattice[lattice.norm(dim=-1) < 1e-3] = torch.inf
        lattice[~atoms.get_pbc()] = torch.inf
        energy = torch.tensor(atoms.get_potential_energy(), dtype=precision)
        forces = torch.from_numpy(atoms.get_forces()).to(precision)

        data = Data()
        data.z = z.reshape(-1)
        data.pos = pos.reshape(-1, 3) * units['length']
        data.lattice = lattice.reshape(1, 3, 3) * units['length']
        data.energy = energy.reshape(1) * units['energy']
        data.force = forces.reshape(-1, 3) * units['energy'] / units['length']

        if pre_filter is not None and not pre_filter(data):
            continue
        if pre_transform is not None:
            data = pre_transform(data)
        data_list.append(data)

    return data_list


class MolecularStatistics(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data):
        stats = {}

        z = data.z.long().cpu()
        z_unique = z.unique()
        # stats['z'] = z_unique

        batch = data.batch.cpu()

        try:
            energy = data.energy.cpu()
            formula = scatter(nn.functional.one_hot(z), batch, dim=0).to(energy.dtype)
            solution = torch.linalg.lstsq(formula, energy, driver='gelsd').solution
            energy_shifts = torch.zeros(118 + 1, dtype=energy.dtype, device=energy.device)
            energy_shifts[z_unique] = solution[z_unique]
            stds = ((energy - torch.matmul(formula, solution)).square().sum() / (formula).sum()).sqrt()
            energy_scale = torch.ones(118 + 1, dtype=energy.dtype, device=energy.device)
            energy_scale[z_unique] = stds
            stats['energy'] = {'shift': energy_shifts, 'scale': energy_scale}
        except AttributeError:
            pass
        try:
            force = data.force.norm(dim=-1).cpu()
            means = scatter(force, z, reduce='mean')
            force_scale = torch.ones(118 + 1, dtype=force.dtype, device=force.device)
            force_scale[z_unique] = means[z_unique]
            stats['force'] = {'scale': force_scale}
        except AttributeError:
            pass
        return stats
