import torch
from torch_geometric.data import Data
from .neighbors import RadiusGraph


class EdgeTransform(RadiusGraph):
    def __init__(self, r, loop=False, flow="source_to_target", num_workers=1):
        super().__init__(r, loop, flow, num_workers)

    def forward(self, data: Data) -> Data:
        # if not hasattr(data, "edge_index"):
        data = super().forward(data)

        # shape (nedges, 1)
        bonds = data.bonds[data.edge_index[0], data.edge_index[1]]
        data.bonds = bonds.unsqueeze(-1)

        return data
