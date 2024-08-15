import os
import os.path as osp
from tqdm import tqdm
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import scatter

from newtonnet.data import RadiusGraph


class MolecularDataset(InMemoryDataset):
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
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        cutoff: float = 5.0,
        precision: torch.dtype = torch.float,
    ) -> None:
        if pre_transform is None:
            pre_transform = RadiusGraph(cutoff)
        self.cutoff = cutoff
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
        return ['data.pt']

    def process(self) -> None:
        data_list = []
        data_path = self.processed_paths[0]
        for raw_path in tqdm(self.raw_paths):
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

class MolecularStatistics(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data):
        z = data.z.long().cpu()
        edge_index = data.edge_index.cpu()
        batch = data.batch.cpu()
        energy = data.energy.cpu()
        force = data.force.norm(dim=-1).cpu()

        z_max = z.max().item()
        node_count = z.size(0)
        edge_count = edge_index.size(1)
        neighbor_count = edge_count / node_count
        formula = scatter(nn.functional.one_hot(z), batch, dim=0).float()
        energy_shifts = torch.linalg.lstsq(formula, energy, driver='gelsd').solution
        energy_shifts[energy_shifts.abs() < 1e-3] = 0
        energy_scale = ((energy - torch.matmul(formula, energy_shifts)).square().sum() / (formula).sum()).sqrt()
        force_scale = scatter(force, z, reduce='mean')
        force_scale[force_scale.abs() < 1e-3] = 0

        stats = {}
        stats['z'] = z.unique()
        stats['average_neighbor_count'] = neighbor_count
        stats['properties'] = {
            'energy': {
                'shift': energy_shifts[:z_max+1],
                'scale': energy_scale,
            },
            'force': {
                'scale': force_scale[:z_max+1],
            },
        }
        return stats
