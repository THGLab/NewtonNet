import os
import os.path as osp
from typing import Callable, List, Optional, Union
import json
from ase import units

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse, scatter

from newtonnet.data import RadiusGraph


class MolecularDataset(InMemoryDataset):
    '''
    This class is a dataset for molecular data.
    
    Args:
        root (str): The root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in a data object and returns a transformed version. The data object will be transformed before every access. Default: None.
        pre_transform (callable, optional): A function/transform that takes in a data object and returns a transformed version. The data object will be transformed before being saved to disk. Default: RadiusGraph().
        pre_filter (callable, optional): A function that takes in a data object and returns a boolean value, indicating whether the data object should be included in the final dataset. Default: None.
        force_reload (bool): Whether to re-process the dataset. Default: False.
        precision (torch.dtype): The precision of the data. Default: torch.float.
    '''
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = RadiusGraph(),
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        precision: torch.dtype = torch.float,
    ) -> None:
        self.precision = precision
        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str]]:
        names = [name for name in os.listdir(self.raw_dir) if name.endswith('.npz')]
        return names

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'stats.json']

    def process(self) -> None:
        data_list = []
        data_path = self.processed_paths[0]
        stats_path = self.processed_paths[1]
        for raw_path in self.raw_paths:
            raw_data = np.load(raw_path)

            z = torch.from_numpy(raw_data['Z']).int()
            pos = torch.from_numpy(raw_data['R']).to(self.precision)
            energy = torch.from_numpy(raw_data['E']).to(self.precision)
            force = torch.from_numpy(raw_data['F']).to(self.precision)

            for i in range(pos.size(0)):
                data = Data(
                    z=z.reshape(-1) if z.dim() < 2 else z[i].reshape(-1),
                    pos=pos[i].reshape(-1, 3),
                    energy=energy[i].reshape(-1),
                    force=force[i].reshape(-1, 3),
                    )
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        self.save(data_list, data_path)
        self.save_stats(data_list, stats_path)

    def save_stats(self, data_list: List[Data], stats_path: str) -> None:
        z_list, formula_list = [], []
        energy_list, force_list = [], []
        for data in data_list:
            z_list.append(data.z)
            formula_list.append(torch.bincount(data.z))
            energy_list.append(data.energy)
            force_list.append(data.force.norm(dim=-1))

        z = torch.cat(z_list, dim=0).long().cpu()
        formula = torch.stack(formula_list, dim=0).float().cpu()
        energy = torch.cat(energy_list, dim=0).cpu()
        force = torch.cat(force_list, dim=0).cpu()
        energy_shifts = torch.linalg.lstsq(formula, energy, driver='gelsd').solution
        energy_shifts[energy_shifts.abs() < 1e-12] = 0
        energy_scale = ((energy - torch.matmul(formula, energy_shifts)).square().sum() / (formula).sum()).sqrt()
        # energy_scale = (energy - torch.matmul(formula, energy_shifts)).std()
        force_scale = scatter(force, z, reduce='mean')

        with open(stats_path, 'w') as f:
            z, energy_shift = dense_to_sparse(energy_shifts.unsqueeze(-1))
            _, force_scale = dense_to_sparse(force_scale.unsqueeze(-1))
            json.dump({'z': z[0].tolist(), 'energy_shift': energy_shift.tolist(), 'energy_scale': energy_scale.item(), 'force_scale': force_scale.tolist()}, f)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'