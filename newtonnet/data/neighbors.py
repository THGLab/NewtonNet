# import torch

# from newtonnet.layers.shells import ShellProvider


# class NeighborEnvironment(object):
#     '''
#     This class finds atomic environments for each atom in the frame.

#     Parameters:
#         cutoff (float): The cutoff radius. Default: 5.0.
#         pbc (bool): Whether to use periodic boundary conditions. Default: False.
#         cell (torch.tensor): The unit cell size. Default: [[0, 0, 0], [0, 0, 0], [0, 0, 0]], i.e. no periodic boundary.

#     Notes:
#         Sparse tensors are not yet supported.
#     '''
#     def __init__(
#             self, 
#             cutoff: float = 5.0,
#             pbc: bool = False,
#             cell: torch.Tensor = torch.zeros(3, 3),
#             ):
#         self.cutoff = cutoff
#         self.shell = ShellProvider(cutoff=cutoff, pbc=pbc, cell=cell)

#     def _check_shapes(self, positions, numbers):
#         assert positions.ndim == 3, 'positions must have 3 dimensions (n_data, n_atoms, 3).'
#         assert numbers.ndim == 2, 'numbers must have 2 dimensions (n_data, n_atoms).'
#         assert positions.shape[0] == numbers.shape[0], 'positions and numbers must have same dimension 0 (n_data).'
#         assert positions.shape[1] == numbers.shape[1], 'positions and numbers must have same dimension 1 (n_atoms).'
#         assert positions.shape[2] == 3, 'positions must have 3 coordinates at dimension 2 (x, y, z).'

#     def get_environment(self, positions, numbers):
#         '''
#         This function finds atomic environments for each atom in the frame.

#         Parameters:
#             positions (torch.tensor): The atomic positions with shape (n_data, n_atoms, 3).
#             numbers (torch.tensor): The atomic numbers with shape (n_data, n_atoms).

#         Returns:
#             distances (torch.tensor): The distances between atoms with shape (n_data, n_atoms, n_atoms).
#             distance_vectors (torch.tensor): The distance vectors between atoms with shape (n_data, n_atoms, n_atoms, 3).
#             atom_mask (torch.tensor): The mask for atoms with shape (n_data, n_atoms).
#             neighbor_mask (torch.tensor): The mask for neighbors with shape (n_data, n_atoms, n_atoms).
#         '''
    
#         _, n_atoms, _ = positions.shape

#         self._check_shapes(positions, numbers)

#         # mask based on zero atomic_numbers
#         atom_mask = (numbers > 0)  # data_size, n_atoms

#         neighbor_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)  # data_size, n_atoms, n_atoms
#         neighbor_mask[:, torch.arange(n_atoms), torch.arange(n_atoms)] = 0

#         # distances
#         distances, distance_vectors, neighbor_mask = self.shell(positions, neighbor_mask)
#         # atom_mask = atom_mask.to_sparse()
#         print(neighbor_mask.dtype)

#         return distances, distance_vectors, atom_mask, neighbor_mask
    
    
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.datasets import MD17
from torch_geometric.loader import DataLoader

class RadiusGraph(BaseTransform):
    r"""Creates edges based on node positions :obj:`data.pos` to all points
    within a given distance (functional name: :obj:`radius_graph`).

    Args:
        r (float): The distance. (default: :obj:`0.5`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`.
            This flag is only needed for CUDA tensors. (default: :obj:`32`)
        flow (str, optional): The flow direction when using in combination with
            message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
    """
    def __init__(
        self,
        r: float = 5.0,
        loop: bool = False,
        max_num_neighbors: int = 32,
        flow: str = 'source_to_target',
        num_workers: int = 1,
    ) -> None:
        self.r = r
        self.loop = loop
        self.max_num_neighbors = max_num_neighbors
        self.flow = flow
        self.num_workers = num_workers

    def forward(self, data: Data) -> Data:
        assert data.pos is not None

        data.edge_index = torch_geometric.nn.radius_graph(
            data.pos,
            self.r,
            data.batch,
            self.loop,
            max_num_neighbors=self.max_num_neighbors,
            flow=self.flow,
            num_workers=self.num_workers,
        )#.sort(dim=0)[0].unique(dim=1)
        data.edge_attr = None

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(r={self.r})'