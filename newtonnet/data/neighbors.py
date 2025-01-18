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
        flow (str, optional): The flow direction when using in combination with
            message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
    """
    def __init__(
        self,
        r: float,
        loop: bool = False,
        flow: str = 'source_to_target',
        num_workers: int = 1,
    ) -> None:
        self.r = r
        self.loop = loop
        self.max_num_neighbors = 1024
        self.flow = flow
        self.num_workers = num_workers

    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        if data.lattice is not None and data.lattice.max(dim=-1).values.isfinite().any():
            n_cell = (self.r / data.lattice.norm(dim=-1)).ceil().int().flatten()
            assert len(n_cell) == 3
            n_cell_tot = (2 * n_cell + 1).prod()
            shift = torch.tensor([[i, j, k] for i in range(-n_cell[0], n_cell[0] + 1) for j in range(-n_cell[1], n_cell[1] + 1) for k in range(-n_cell[2], n_cell[2] + 1)], dtype=data.pos.dtype, device=data.pos.device)
            shift = shift @ data.lattice
            shift = shift.nan_to_num()
            shifted_pos = data.pos[:, None, :] + shift  # shape: (n_node, n_cell_tot, 3)
            shifted_pos = shifted_pos.reshape(-1, 3)  # shape: (n_node * n_cell_tot, 3)
            shifted_node_index = torch.arange(data.pos.shape[0], dtype=torch.long, device=data.pos.device)[:, None].repeat(1, n_cell_tot)  # shape: (n_node, n_cell_tot)
            shifted_node_index = shifted_node_index.reshape(-1)  # shape: (n_node * n_cell_tot)
            shifted_node_isoriginal = torch.zeros(data.pos.shape[0], n_cell_tot, dtype=torch.bool, device=data.pos.device)  # shape: (n_node, n_cell_tot)
            shifted_node_isoriginal[:, n_cell_tot // 2] = True
            shifted_node_isoriginal = shifted_node_isoriginal.reshape(-1)  # shape: (n_node * n_cell_tot)
            if data.batch is not None:
                shifted_batch = data.batch[:, None].repeat(1, n_cell_tot)  # shape: (n_node, n_cell_tot)
                shifted_batch = shifted_batch.reshape(-1)  # shape: (n_node * n_cell_tot)
            else:
                shifted_batch = None
            shifted_edge_index = radius_graph(
                shifted_pos,
                self.r,
                shifted_batch,
                self.loop,
                max_num_neighbors=self.max_num_neighbors,
                flow=self.flow,
                num_workers=self.num_workers,
            )#.sort(dim=0)[0].unique(dim=1)
            shifted_edge_isoriginal = shifted_node_isoriginal[shifted_edge_index[0]]
            shifted_edge_index = shifted_edge_index[:, shifted_edge_isoriginal]
            data.edge_index = shifted_node_index[shifted_edge_index]
            data.disp = shifted_pos[shifted_edge_index[0]] - shifted_pos[shifted_edge_index[1]]
        else:
            data.edge_index = radius_graph(
                data.pos,
                self.r,
                data.batch,
                self.loop,
                max_num_neighbors=self.max_num_neighbors,
                flow=self.flow,
                num_workers=self.num_workers,
            )#.sort(dim=0)[0].unique(dim=1)
            data.disp = data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]]

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(r={self.r})'