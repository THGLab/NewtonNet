import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.nn import radius_graph

class RadiusGraph(BaseTransform):
    r"""Creates edges based on node positions :obj:`data.pos` to all points
    within a given distance (functional name: :obj:`radius_graph`).

    Args:
        r (float): The distance.
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
    """
    def __init__(
        self,
        r: float,
        loop: bool = False,
    ) -> None:
        self.r = r
        self.loop = loop

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        
        # Create full graph
        n_node = data.pos.shape[0]
        row = torch.arange(n_node, device=data.pos.device).view(n_node, 1).repeat(1, n_node).view(-1)
        col = torch.arange(n_node, device=data.pos.device).view(n_node, 1).repeat(n_node, 1).view(-1)
        edge_index = torch.stack([row, col], dim=0)
        if data.batch is not None:
            edge_index = edge_index[:, data.batch[row] == data.batch[col]]
        if not self.loop:
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]
            

        # Compute distances
        disp = data.pos[edge_index[0]] - data.pos[edge_index[1]]
        if data.cell is not None and not (data.cell == 0).all():
            if data.batch is not None:
                cell = data.cell[data.batch]
            else:
                cell = data.cell.repeat(n_node, 1, 1)
            cell = cell[edge_index[0]]
            scaled_disp = torch.linalg.solve(cell.transpose(1, 2), disp)
            disp = disp - torch.bmm(cell, torch.round(scaled_disp).unsqueeze(-1)).squeeze(-1)

        # Filter edges based on distance
        mask = (disp.norm(dim=1) < self.r)
        edge_index = edge_index[:, mask]
        disp = disp[mask]

        # Record to data
        data.edge_index = edge_index
        data.disp = disp

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(r={self.r})'