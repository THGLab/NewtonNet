import os
import os.path as osp
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data

from newtonnet.data.neighbors import RadiusGraph


class MolecularDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:

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
        processed_path = self.processed_paths[0]
        for raw_path in self.raw_paths:
            raw_data = np.load(raw_path)

            z = torch.from_numpy(raw_data['Z']).int()
            pos = torch.from_numpy(raw_data['R']).float()
            energy = torch.from_numpy(raw_data['E']).float()
            force = torch.from_numpy(raw_data['F']).float()

            for i in range(pos.size(0)):
                data = Data(z=z, pos=pos[i], energy=energy[i], force=force[i])
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        self.save(data_list, processed_path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}')"