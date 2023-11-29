import torch
from torch import nn
from torch.autograd import grad
from torch.jit.annotations import Optional

# from newtonnet.layers import Dense
from newtonnet.layers.shells import ShellProvider
from newtonnet.layers.scalers import ScaleShift, TrainableScaleShift
from newtonnet.layers.cutoff import CosineCutoff, PolynomialCutoff
from newtonnet.layers.representations import RadialBesselLayer


class NewtonNet(nn.Module):
    """
    Molecular Newtonian Message Passing

    Parameters
    ----------
    resolution: int
        number of radial functions to describe interatomic distances

    n_features: int
        number of neurons in the latent layer. This number will remain fixed in the entire network except
        for the last fully connected network that predicts atomic energies.

    activation: function
        activation function from newtonnet.layers.activations
        you can aslo get it by string from newtonnet.layers.activations.get_activation_by_string

    n_interactions: int, default: 3
        number of interaction blocks

    dropout: float, default: 0.0
        dropout rate
    
    max_z: int, default: 10
        maximum atomic number Z in the dataset

    cutoff: float, default: 5.0
        cutoff radius in Angstrom

    cutoff_network: str, default: 'poly'
        cutoff function, can be 'poly' or 'cosine'

    normalizer: tuple, default: (0.0, 1.0)
        mean and standard deviation of the target property. If you have a dictionary of normalizers for each atomic type,
        you can pass it as a dictionary. For example, {'1': (0.0, 1.0), '6': (0.0, 1.0), '7': (0.0, 1.0), '8': (0.0, 1.0)}

    normalize_atomic: bool, default: False
        whether to normalize the atomic energies

    requires_dr: bool, default: False
        whether to compute the forces

    device: torch.device, default: None
        device to run the network

    create_graph: bool, default: False
        whether to create the graph for the gradient computation

    shared_interactions: bool, default: False
        whether to share the interaction block weights

    return_latent: bool, default: False
        whether to return the latent forces
    """

    def __init__(
            self,
            n_basis,
            n_features,
            activation,
            n_layers=3,
            dropout=0.0,
            max_z=10,
            cutoff=5.0,
            cutoff_network='poly',
            normalizer=(0.0, 1.0),
            normalize_atomic=False,
            requires_dr=False,
            device=None,
            create_graph=False,
            share_layers=False,
            return_hessian=False,
            layer_norm=False,
            atomic_properties_only=False,
            double_update_latent=True,
            pbc=False,
            aggregration='sum'):

        super(NewtonNet, self).__init__()

        self.requires_dr = requires_dr
        self.create_graph = create_graph
        self.normalize_atomic = normalize_atomic
        self.return_hessian = return_hessian
        self.pbc = pbc
        self.epsilon = 1.0e-8

        # test

        shell_cutoff = None
        if pbc:
            # make the cutoff here a little bit larger so that it can be handled with differentiable cutoff layer in interaction block
            shell_cutoff = cutoff * 1.1

        self.shell = ShellProvider(periodic_boundary=pbc, cutoff=shell_cutoff)
        self.distance_expansion = RadialBesselLayer(
            n_basis, cutoff, device=device
        )

        # atomic embedding
        self.n_features = n_features
        self.embedding = nn.Embedding(max_z, n_features, padding_idx=0)

        # d1 message
        self.n_interactions = n_layers
        if share_layers:
            # use the same message instance (hence the same weights)
            self.dycalc = nn.ModuleList(
                [
                    DynamicsCalculator(
                        n_features=n_features,
                        resolution=n_basis,
                        activation=activation,
                        cutoff=cutoff,
                        cutoff_network=cutoff_network,
                        double_update_latent=double_update_latent,
                        layer_norm=layer_norm,
                    )
                ]
                * n_layers
            )
        else:
            # use one SchNetInteraction instance for each interaction
            self.dycalc = nn.ModuleList(
                [
                    DynamicsCalculator(
                        n_features=n_features,
                        resolution=n_basis,
                        activation=activation,
                        cutoff=cutoff,
                        cutoff_network=cutoff_network,
                        double_update_latent=double_update_latent,
                        layer_norm=layer_norm,
                    )
                    for _ in range(n_layers)
                ]
            )

        # layer norm
        # self.layer_norm = layer_norm
        # if layer_norm:
        #     self.norm = nn.ModuleList([nn.LayerNorm(n_features) for _ in range(n_interactions)])

        # final dense network
        self.atomic_energy = AtomicEnergy(n_features, activation, dropout)

        self.normalize_atomic = normalize_atomic
        if normalize_atomic:
            self.inverse_normalize = TrainableScaleShift(max_z)
        else:
            if type(normalizer) is dict:
                self.inverse_normalize = nn.ModuleDict({
                    str(atom_num): ScaleShift(
                        mean=torch.tensor(normalizer[atom_num][0], device=device),
                        stddev=torch.tensor(normalizer[atom_num][1], device=device),
                        ) for atom_num in normalizer
                    })
            else:
                self.inverse_normalize = nn.ModuleDict({
                    'all': ScaleShift(
                        mean=torch.tensor(normalizer[0], device=device),
                        stddev=torch.tensor(normalizer[1], device=device),
                        )
                    })
                # self.inverse_normalize = ScaleShift(
                #     mean=torch.tensor(normalizer[0], device=device),
                #     stddev=torch.tensor(normalizer[1], device=device),
                #     )

        self.atomic_properties_only = atomic_properties_only
        self.aggregration = aggregration


    def forward(
            self,
            atomic_numbers: torch.Tensor,
            positions: torch.Tensor,
            atom_mask: torch.Tensor,
            neighbors: torch.Tensor,
            neighbor_mask: torch.Tensor,
            distances: torch.Tensor,
            distance_vectors: torch.Tensor,
            lattice: torch.Tensor = torch.eye(3),
            ):

        # initiate main containers
        a = self.embedding(atomic_numbers)  # B,A,nf
        f_dir = torch.zeros_like(positions)  # B,A,3
        f_dynamics = torch.zeros(positions.size() + (self.n_features,), device=positions.device)  # B,A,3,nf
        r_dynamics = torch.zeros(positions.size() + (self.n_features,), device=positions.device)  # B,A,3,nf
        e_dynamics = torch.zeros_like(a)  # B,A,nf

        # require grad
        if self.requires_dr:
            positions.requires_grad_()

        # compute distances (B,A,N) and distance vectors (B,A,N,3)
        if self.requires_dr:
            distances, distance_vectors, neighbors, neighbor_mask = self.shell(positions, neighbors, neighbor_mask, lattice)
            distance_vectors = distance_vectors / (distances[:, :, :, None] + self.epsilon)

        # comput d1 representation (B, A, N, G)
        rbf = self.distance_expansion(distances)

        # compute interaction block and update atomic embeddings
        # for i_interax in range(self.n_interactions):
        for dynamics_calculator in self.dycalc:
            # print('iter: ', i_interax)

            # messages
            a, f_dir, f_dynamics, r_dynamics, e_dynamics = dynamics_calculator(
                a, rbf, distances, distance_vectors, neighbors, neighbor_mask, f_dir, f_dynamics, r_dynamics, e_dynamics
                )  # B,A,f  # B,A,N,f

            # if self.layer_norm:
            #     a = self.norm[i_interax](a)

        # When using the network to obtain atomic properties only
        if self.atomic_properties_only:
            Ai = self.atomic_energy(a)
            if self.normalize_atomic:
                Ai = self.inverse_normalize(Ai, atomic_numbers)
            elif hasattr(self.inverse_normalize, 'keys') and hasattr(self.inverse_normalize, 'values'):
                for atomic_type in self.inverse_normalize:
                    if atomic_type == 'all':
                        atomic_filter = atomic_numbers > 0
                    else:
                        atomic_filter = atomic_numbers == int(atomic_type)
                    Ai[atomic_filter] = self.inverse_normalize[atomic_type](Ai[atomic_filter])
            return {'Ai': Ai}

        # output net
        Ei = self.atomic_energy(a)
        if self.normalize_atomic:
            Ei = self.inverse_normalize(Ei, atomic_numbers)
        elif hasattr(self.inverse_normalize, 'keys') and hasattr(self.inverse_normalize, 'values'):
            for atomic_type in self.inverse_normalize:
                if atomic_type == 'all':
                    atomic_filter = atomic_numbers > 0
                else:
                    atomic_filter = atomic_numbers == int(atomic_type)
                Ei[atomic_filter] = self.inverse_normalize[atomic_type](Ei[atomic_filter])

        # inverse normalize
        Ei = Ei * atom_mask[..., None]  # (B,A,1)
        if self.aggregration == 'sum':
            E = torch.sum(Ei, 1)  # (B,1)
        elif self.aggregration == 'mean':
            E = torch.mean(Ei, 1)
        elif self.aggregration == 'max':
            E = torch.max(Ei, 1).values
        else:
            raise ValueError('Unknown aggregration method: {}'.format(self.aggregration))
        # if not self.normalize_atomic:
            # E = self.inverse_normalize(E)

        # if self.return_hessian:
        #     return E

        if self.requires_dr:
            if self.return_hessian:
                dE = grad([E], [positions], grad_outputs=[torch.ones(E.shape[0], 1, device=positions.device) if E.shape[0]==positions.shape[0] else None], create_graph=True, retain_graph=True)[0]
                # TODO: make Hessian calculations work
                # ddE = torch.zeros(E.shape[0], R.shape[1], R.shape[2], R.shape[1], R.shape[2], device=R.device)
                # for A_ in range(R.shape[1]):
                #     for X_ in range(R.shape[2]):
                #         dE[:, A_, X_]
                #         ddE[:, A_, X_, :, :] = grad(dE[:, A_, X_], R, grad_outputs=torch.ones(E.shape[0], device=R.device), create_graph=False, retain_graph=True)[0]
                # ddE = torch.stack([grad(dE, R, grad_outputs=V, create_graph=True, retain_graph=True, allow_unused=True)[0] for V in torch.eye(R.shape[1] * R.shape[2], device=R.device).reshape((-1, 1, R.shape[1], R.shape[2])).repeat(1, R.shape[0], 1, 1)])
                # ddE = torch.vmap(lambda V: grad(dE, R, grad_outputs=V, create_graph=True, retain_graph=True))(torch.eye(R.shape[1] * R.shape[2], device=R.device).reshape((-1, 1, R.shape[1], R.shape[2])).repeat(1, R.shape[0], 1, 1))
                # ddE = ddE.permute(1,2,3,0).unflatten(dim=3, sizes=(-1, 3))
            else:
                dE = grad([E], [positions], grad_outputs=[torch.ones(E.shape[0], 1, device=positions.device) if E.shape[0]==positions.shape[0] else None], create_graph=True, retain_graph=True)[0]
        else:
            # dE = data['F']
            dE = torch.zeros_like(positions)
        assert dE is not None
        dE = -dE

        if self.return_hessian:
            # TODO: make Hessian calculations work
            # return {'R': R, 'E': E, 'F': dE, 'H': ddE, 'Ei': Ei, 'F_latent': f_dir}
            return {'R': positions, 'E': E, 'F': dE, 'Ei': Ei, 'F_latent': f_dir}
        else:
            return {'R': positions, 'E': E, 'F': dE, 'Ei': Ei, 'F_latent': f_dir}


class DynamicsCalculator(nn.Module):

    def __init__(
            self,
            n_features,
            resolution,
            activation,
            cutoff,
            cutoff_network,
            double_update_latent=True,
            epsilon=1e-8,
            layer_norm=False,
    ):
        super(DynamicsCalculator, self).__init__()

        self.n_features = n_features
        self.epsilon = epsilon

        # non-directional message passing
        # self.phi_rbf = Dense(resolution, n_features, activation=None)
        self.phi_rbf = nn.Linear(resolution, n_features)

        self.phi_a = nn.Sequential(
            # Dense(n_features, n_features, activation=activation),
            # Dense(n_features, n_features, activation=None),
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )

        # cutoff layer used in interaction block
        self.cutoff_network = PolynomialCutoff(cutoff, degree=9)

        # directional message passing
        # self.phi_f = Dense(n_features, 1, activation=None, bias=False)
        self.phi_f = nn.Linear(n_features, 1, bias=False)
        self.phi_f_scale = nn.Sequential(
            # Dense(n_features, n_features, activation=activation),
            # Dense(n_features, n_features, activation=None),
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )
        self.phi_r = nn.Sequential(
            # Dense(n_features, n_features, activation=None),
            nn.Linear(n_features, n_features),
        )
        self.phi_r = nn.Sequential(
            # Dense(n_features, n_features, activation=activation, xavier_init_gain=0.001),
            # Dense(n_features, n_features, activation=None),
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )
        self.phi_r_ext = nn.Sequential(
            # Dense(n_features, n_features, activation=activation, bias=False),
            # Dense(n_features, n_features, activation=None, bias=False),
            nn.Linear(n_features, n_features, bias=False),
            activation,
            nn.Linear(n_features, n_features, bias=False),
        )

        self.phi_e = nn.Sequential(
            # Dense(n_features, n_features, activation=activation),
            # Dense(n_features, n_features, activation=None)
            nn.Linear(n_features, n_features),
            activation,
            nn.Linear(n_features, n_features),
        )

        self.double_update_latent = double_update_latent

        self.layer_norm = layer_norm
        self.norm = nn.LayerNorm(n_features)

    def gather_neighbors(self, inputs, N):
        n_features = inputs.size()[-1]
        n_dim = inputs.dim()
        b, a, n = N.size()  # batch, atoms, neighbors size

        if n_dim == 3:
            N = N.view(-1, a * n, 1)  # B,A*N,1
            N = N.expand(-1, -1, n_features)
            out = torch.gather(inputs, dim=1, index=N)
            return out.view(b, a, n, n_features)  # B,A,N,n_features

        elif n_dim == 4:
            N = N.view(-1, a * n, 1, 1)  # B,A*N,1,1
            N = N.expand(-1, -1, 3, n_features)
            out = torch.gather(inputs, dim=1, index=N)
            return out.view(b, a, n, 3, n_features)  # B,A,N,3,n_features

    def sum_neighbors(self, x, mask, dim:int=2, avg:bool=False):
        """

        Parameters
        ----------
        x: torch.tensor
            usually of shape B,A,N,nf
        mask: torch.tensor
            usually of shape B,A,N
        dim: int
            the dimension to sum

        avg: bool
            if True, returns the average output by dividing the sum by number of neighbors.

        Returns
        -------

        """
        dim_diff = x.dim() - mask.dim()
        for _ in range(dim_diff):
            mask = mask.unsqueeze(-1)

        x = x * mask
        out = torch.sum(x, dim=dim)

        if avg:
            n_atoms = torch.sum(mask, dim)
            n_atoms = torch.max(n_atoms, other=torch.ones_like(n_atoms))
            out = out / n_atoms

        return out

    def forward(self, a, rbf, distances, distance_vector, N, NM,
                f_dir, f_dynamics, r_dynamics, e_dynamics
                ):

        # map decomposed distances
        rbf_msij = self.phi_rbf(rbf)  # B,A,N,nf

        # cutoff
        C = self.cutoff_network(distances)
        rbf_msij = rbf_msij * C.unsqueeze(-1)

        # map atomic features
        a_msij = self.phi_a(a)  # B,A,3*nf

        # copy central atom features for the element-wise multiplication
        ai_msij = a_msij.repeat(1, 1, rbf_msij.size(2))
        ai_msij = ai_msij.view(rbf_msij.size())  # B,A,N,nf

        # look up neighboring atoms features based on the schnet contiuous filter implementation
        aj_msij = self.gather_neighbors(a_msij, N)  # B,A,N,nf
        assert aj_msij is not None

        # symmetric feature multiplication
        mij = rbf_msij * aj_msij
        msij = mij * ai_msij

        # update a with invariance
        if self.double_update_latent:
            a = a + self.sum_neighbors(msij, NM, dim=2)

        # Dynamics: Forces
        # print('msij:', msij.shape, msij[0,0])
        F_ij = self.phi_f(msij) * distance_vector  # B,A,N,3
        F_i_dir = self.sum_neighbors(F_ij, NM, dim=2)  # B,A,3
        f_dir = f_dir + F_i_dir

        F_ij = self.phi_f_scale(msij).unsqueeze(-2) * F_ij.unsqueeze(-1)  # B,A,N,3,nf
        # print('F_ij:', F_ij.shape, F_ij[0,0])
        F_i = self.sum_neighbors(F_ij, NM, dim=2)  # B,A,3,nf

        # dr
        dr_i = self.phi_r(a).unsqueeze(-2) * F_i  # B,A,3,nf

        dr_j = self.gather_neighbors(r_dynamics, N)  # B,A,N,3,nf
        assert dr_j is not None
        dr_j = self.phi_r_ext(msij).unsqueeze(-2) * dr_j  # B,A,N,3,nf
        # print('dr_j:', dr_j.shape, dr_j[0,0])
        dr_ext = self.sum_neighbors(dr_j, NM, dim=2, avg=False)  # B,A,3,nf

        # update
        f_dynamics = f_dynamics + F_i
        r_dynamics = r_dynamics + dr_i + dr_ext

        # update energy
        de_i = -1.0 * torch.sum(f_dynamics * r_dynamics, dim=-2)  # B,A,nf
        de_i = self.phi_e(a) * de_i
        a = a + de_i
        e_dynamics = e_dynamics + de_i

        # layer norm
        if self.layer_norm:
            a = self.norm(a)

        return a, f_dir, f_dynamics, r_dynamics, e_dynamics


class AtomicEnergy(nn.Module):

    def __init__(self, n_features, activation, dropout):
        super(AtomicEnergy, self).__init__()
        self.environment = nn.Sequential(
            # Dense(n_features, 128, activation=activation, dropout=dropout, norm=False),
            # Dense(128, 64, activation=activation, dropout=dropout, norm=False),
            # Dense(64, 1, activation=None, dropout=0.0, norm=False),
            nn.Linear(n_features, 128),
            activation,
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            activation,
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, a):
        # update atomic features
        out = self.environment(a)

        return out

# import torch
# from torch import nn
# from torch.autograd import grad
# from torch.jit.annotations import Optional

# # from newtonnet.layers import Dense
# from newtonnet.layers.shells import ShellProvider
# from newtonnet.layers.scalers import ScaleShift, TrainableScaleShift
# from newtonnet.layers.cutoff import CosineCutoff, PolynomialCutoff
# from newtonnet.layers.representations import RadialBesselLayer


# class NewtonNet(nn.Module):
#     """
#     Molecular Newtonian Message Passing

#     Parameters
#     ----------
#     resolution: int
#         number of radial functions to describe interatomic distances

#     n_features: int
#         number of neurons in the latent layer. This number will remain fixed in the entire network except
#         for the last fully connected network that predicts atomic energies.

#     activation: function
#         activation function from newtonnet.layers.activations
#         you can aslo get it by string from newtonnet.layers.activations.get_activation_by_string

#     n_interactions: int, default: 3
#         number of interaction blocks

#     dropout: float, default: 0.0
#         dropout rate
    
#     max_z: int, default: 10
#         maximum atomic number Z in the dataset

#     cutoff: float, default: 5.0
#         cutoff radius in Angstrom

#     cutoff_network: str, default: 'poly'
#         cutoff function, can be 'poly' or 'cosine'

#     normalizer: tuple, default: (0.0, 1.0)
#         mean and standard deviation of the target property. If you have a dictionary of normalizers for each atomic type,
#         you can pass it as a dictionary. For example, {'1': (0.0, 1.0), '6': (0.0, 1.0), '7': (0.0, 1.0), '8': (0.0, 1.0)}

#     normalize_atomic: bool, default: False
#         whether to normalize the atomic energies

#     requires_dr: bool, default: False
#         whether to compute the forces

#     device: torch.device, default: None
#         device to run the network

#     create_graph: bool, default: False
#         whether to create the graph for the gradient computation

#     shared_interactions: bool, default: False
#         whether to share the interaction block weights

#     return_latent: bool, default: False
#         whether to return the latent forces
#     """

#     def __init__(
#             self,
#             n_basis: int = 20,
#             n_features: int = 128,
#             activation: nn.Module = nn.SiLU(),
#             n_layers: int = 3,
#             dropout: float = 0.0,
#             max_z: int = 10,
#             cutoff: float = 5.0,
#             cutoff_network: nn.Module = PolynomialCutoff(),
#             normalizer: tuple = (0.0, 1.0),
#             normalize_atomic: bool = False,
#             requires_dr: bool = False,
#             device: torch.device = None,
#             create_graph: bool = False,
#             share_layers: bool = False,
#             return_hessian: bool = False,
#             layer_norm: bool = False,
#             atomic_properties_only: bool = False,
#             double_update_latent: bool = True,
#             period_boundary: bool = False,
#             lattice: Optional[torch.Tensor] = None,
#             aggregration: str = 'sum',
#     ) -> None:

#         super(NewtonNet, self).__init__()

#         self.requires_dr = requires_dr
#         self.create_graph = create_graph
#         self.normalize_atomic = normalize_atomic
#         self.return_hessian = return_hessian
#         self.period_boundary = period_boundary

#         # atomic embedding
#         self.n_features = n_features
#         self.node_embedding = nn.Embedding(max_z, n_features, padding_idx=0)

#         # edge embedding
#         shell_cutoff = None
#         if period_boundary:
#             # make the cutoff here a little bit larger so that it can be handled with differentiable cutoff layer in interaction block
#             shell_cutoff = cutoff * 1.1
#         self.shell = ShellProvider(period_boundary=period_boundary, cutoff=shell_cutoff)
#         self.edge_embedding = RadialBesselLayer(n_basis, cutoff, device=device)
#         self.epsilon = 1e-8

#         # d1 message
#         self.n_layers = n_layers
#         if share_layers:
#             # use the same message instance (hence the same weights)
#             self.dycalc = nn.ModuleList(
#                 [
#                     DynamicsCalculator(
#                         n_features=n_features,
#                         n_basis=n_basis,
#                         activation=activation,
#                         cutoff_network=cutoff_network,
#                         double_update_latent=double_update_latent,
#                         layer_norm=layer_norm,
#                     )
#                 ]
#                 * n_layers
#             )
#         else:
#             # use one SchNetInteraction instance for each interaction
#             self.dycalc = nn.ModuleList(
#                 [
#                     DynamicsCalculator(
#                         n_features=n_features,
#                         n_basis=n_basis,
#                         activation=activation,
#                         cutoff_network=cutoff_network,
#                         double_update_latent=double_update_latent,
#                         layer_norm=layer_norm,
#                     )
#                     for _ in range(n_layers)
#                 ]
#             )

#         # layer norm
#         # self.layer_norm = layer_norm
#         # if layer_norm:
#         #     self.norm = nn.ModuleList([nn.LayerNorm(n_features) for _ in range(n_interactions)])

#         # final dense network
#         self.atomic_energy = AtomicEnergy(n_features, activation, dropout)

#         self.normalize_atomic = normalize_atomic
#         if normalize_atomic:
#             self.inverse_normalize = TrainableScaleShift(max_z)
#         else:
#             if type(normalizer) is dict:
#                 self.inverse_normalize = nn.ModuleDict({
#                     str(atom_num): ScaleShift(
#                         mean=torch.tensor(normalizer[atom_num][0], device=device),
#                         stddev=torch.tensor(normalizer[atom_num][1], device=device),
#                         ) for atom_num in normalizer
#                     })
#             else:
#                 self.inverse_normalize = nn.ModuleDict({
#                     'all': ScaleShift(
#                         mean=torch.tensor(normalizer[0], device=device),
#                         stddev=torch.tensor(normalizer[1], device=device),
#                         )
#                     })
#                 # self.inverse_normalize = ScaleShift(
#                 #     mean=torch.tensor(normalizer[0], device=device),
#                 #     stddev=torch.tensor(normalizer[1], device=device),
#                 #     )

#         self.atomic_properties_only = atomic_properties_only
#         self.aggregration = aggregration

#     def forward(
#             self, 
#             atomic_numbers: torch.Tensor,
#             positions: torch.Tensor,
#             atom_mask: torch.Tensor,
#             neighbors: torch.Tensor,
#             neighbor_mask: torch.Tensor,
#             lattice: Optional[torch.Tensor] = None,
#     ):

#         # initiate main containers
#         invariant_node = self.node_embedding(atomic_numbers)  # batch_size, n_atoms, n_features
#         f_dir = torch.zeros_like(positions)  # batch_size, n_atoms, 3
#         equivariant_node_f = torch.zeros(positions.size() + (self.n_features,), device=positions.device)  # batch_size, n_atoms, 3, n_features
#         equivariant_node_dr = torch.zeros(positions.size() + (self.n_features,), device=positions.device)  # batch_size, n_atoms, 3, n_features
#         e_dynamics = torch.zeros_like(invariant_node)  # batch_size, n_atoms, n_features

#         # require grad
#         if self.requires_dr:
#             positions.requires_grad_()

#         # compute distances (batch_size, n_atoms, n_neighbors, 1) and distance vectors (batch_size, n_atoms, n_neighbors, 3)
#         distances, distance_vector, neighbors, neighbor_mask = self.shell(positions, neighbors, neighbor_mask, lattice)
#         distance_vector = distance_vector / (distances[:, :, :, None] + self.epsilon)


#         # comput d1 representation (batch_size, n_atoms, n_neighbors, n_basis)
#         invariant_edges = self.edge_embedding(distances)

#         # compute interaction block and update atomic embeddings
#         for dynamics_calculator in self.dycalc:
#             # messages
#             invariant_node, f_dir, equivariant_node_f, equivariant_node_dr, e_dynamics = dynamics_calculator(
#                 invariant_node, invariant_edges, distances, distance_vector, neighbors, neighbor_mask, f_dir, equivariant_node_f, equivariant_node_dr, e_dynamics
#                 )

#             # if self.layer_norm:
#             #     a = self.norm[i_interax](a)

#         # When using the network to obtain atomic properties only
#         if self.atomic_properties_only:
#             Ai = self.atomic_energy(invariant_node)
#             if self.normalize_atomic:
#                 Ai = self.inverse_normalize(Ai, atomic_numbers)
#             elif hasattr(self.inverse_normalize, 'keys') and hasattr(self.inverse_normalize, 'values'):
#                 for atomic_type in self.inverse_normalize:
#                     if atomic_type == 'all':
#                         atomic_filter = atomic_numbers > 0
#                     else:
#                         atomic_filter = atomic_numbers == int(atomic_type)
#                     Ai[atomic_filter] = self.inverse_normalize[atomic_type](Ai[atomic_filter])
#             return {'Ai': Ai}

#         # output net
#         node_output = self.atomic_energy(invariant_node)
#         if self.normalize_atomic:
#             node_output = self.inverse_normalize(node_output, atomic_numbers)
#         elif hasattr(self.inverse_normalize, 'keys') and hasattr(self.inverse_normalize, 'values'):
#             for atomic_type in self.inverse_normalize:
#                 if atomic_type == 'all':
#                     atomic_filter = atomic_numbers > 0
#                 else:
#                     atomic_filter = atomic_numbers == int(atomic_type)
#                 node_output[atomic_filter] = self.inverse_normalize[atomic_type](node_output[atomic_filter])

#         # inverse normalize
#         node_output = node_output * atom_mask[..., None]  # (B,A,1)
#         if self.aggregration == 'sum':
#             graph_output = torch.sum(node_output, 1)  # (B,1)
#         elif self.aggregration == 'mean':
#             graph_output = torch.mean(node_output, 1)
#         elif self.aggregration == 'max':
#             graph_output = torch.max(node_output, 1).values
#         else:
#             raise ValueError('Unknown aggregration method: {}'.format(self.aggregration))
#         # if not self.normalize_atomic:
#             # E = self.inverse_normalize(E)

#         # if self.return_hessian:
#         #     return E

#         if self.requires_dr:
#             if self.return_hessian:
#                 dE = grad([graph_output], [positions], grad_outputs=[torch.ones(graph_output.shape[0], 1, device=positions.device) if graph_output.shape[0]==positions.shape[0] else None], create_graph=True, retain_graph=True)[0]
#                 # TODO: make Hessian calculations work
#                 # ddE = torch.zeros(E.shape[0], R.shape[1], R.shape[2], R.shape[1], R.shape[2], device=R.device)
#                 # for A_ in range(R.shape[1]):
#                 #     for X_ in range(R.shape[2]):
#                 #         dE[:, A_, X_]
#                 #         ddE[:, A_, X_, :, :] = grad(dE[:, A_, X_], R, grad_outputs=torch.ones(E.shape[0], device=R.device), create_graph=False, retain_graph=True)[0]
#                 # ddE = torch.stack([grad(dE, R, grad_outputs=V, create_graph=True, retain_graph=True, allow_unused=True)[0] for V in torch.eye(R.shape[1] * R.shape[2], device=R.device).reshape((-1, 1, R.shape[1], R.shape[2])).repeat(1, R.shape[0], 1, 1)])
#                 # ddE = torch.vmap(lambda V: grad(dE, R, grad_outputs=V, create_graph=True, retain_graph=True))(torch.eye(R.shape[1] * R.shape[2], device=R.device).reshape((-1, 1, R.shape[1], R.shape[2])).repeat(1, R.shape[0], 1, 1))
#                 # ddE = ddE.permute(1,2,3,0).unflatten(dim=3, sizes=(-1, 3))
#             else:
#                 dE = grad([graph_output], [positions], grad_outputs=[torch.ones(graph_output.shape[0], 1, device=positions.device) if graph_output.shape[0]==positions.shape[0] else None], create_graph=True, retain_graph=True)[0]
#         else:
#             # dE = data['F']
#             dE = torch.zeros_like(positions)
#         assert dE is not None
#         dE = -dE

#         if self.return_hessian:
#             # TODO: make Hessian calculations work
#             # return {'R': R, 'E': E, 'F': dE, 'H': ddE, 'Ei': Ei, 'F_latent': f_dir}
#             return {'R': positions, 'E': graph_output, 'F': dE, 'Ei': node_output, 'F_latent': f_dir}
#         else:
#             return {'R': positions, 'E': graph_output, 'F': dE, 'Ei': node_output, 'F_latent': f_dir}


# class DynamicsCalculator(nn.Module):

#     def __init__(
#             self,
#             n_features: int = 128,
#             n_basis: int = 20,
#             activation: nn.Module = nn.SiLU(),
#             cutoff_network: nn.Module = PolynomialCutoff(),
#             double_update_latent: bool = True,
#             layer_norm: bool = False,
#     ):
        
#         super(DynamicsCalculator, self).__init__()

#         self.n_features = n_features

#         # non-directional message passing
#         self.calculate_edge_message = nn.Linear(n_basis, n_features)

#         self.calculate_node_message = nn.Sequential(
#             nn.Linear(n_features, n_features),
#             activation,
#             nn.Linear(n_features, n_features),
#         )
#         self.cutoff_network = cutoff_network

#         # directional message passing
#         self.calculate_equivariant_message_coefficient = nn.Linear(n_features, 1, bias=False)
#         self.calculate_equivariant_message_feature = nn.Sequential(
#             nn.Linear(n_features, n_features),
#             activation,
#             nn.Linear(n_features, n_features),
#         )
#         self.phi_r = nn.Sequential(
#             nn.Linear(n_features, n_features),
#         )
#         self.phi_r = nn.Sequential(
#             nn.Linear(n_features, n_features),
#             activation,
#             nn.Linear(n_features, n_features),
#         )
#         self.phi_r_ext = nn.Sequential(
#             nn.Linear(n_features, n_features, bias=False),
#             activation,
#             nn.Linear(n_features, n_features, bias=False),
#         )

#         self.phi_e = nn.Sequential(
#             nn.Linear(n_features, n_features),
#             activation,
#             nn.Linear(n_features, n_features),
#         )

#         self.double_update_latent = double_update_latent

#         self.layer_norm = layer_norm
#         self.norm = nn.LayerNorm(n_features)

#     def gather_neighbors(self, inputs, neighbors):
#         n_features = inputs.size()[-1]
#         n_dim = inputs.dim()
#         batch_size, n_atoms, n_neighbors = neighbors.size()  # batch, atoms, neighbors size

#         if n_dim == 3:    # inputs: batch_size, n_atoms, n_features
#             neighbors = neighbors[:, :, :, None].expand(-1, -1, -1, n_features)    # batch_size, n_atoms, n_neighbors, n_features
#             inputs = inputs[:, :, None, :].expand(-1, -1, n_neighbors, -1)    # batch_size, n_atoms, n_neighbors, n_features
#             outputs = torch.gather(inputs, dim=1, index=neighbors)    # batch_size, n_atoms, n_neighbors, n_features
#             return outputs
#         elif n_dim == 4:    # inputs: batch_size, n_atoms, 3, n_features
#             neighbors = neighbors[:, :, :, None, None].expand(-1, -1, -1, 3, n_features)    # batch_size, n_atoms, n_neighbors, 3, n_features
#             inputs = inputs[:, :, None, :, :].expand(-1, -1, n_neighbors, -1, -1)    # batch_size, n_atoms, n_neighbors, 3, n_features
#             outputs = torch.gather(inputs, dim=1, index=neighbors)    # batch_size, n_atoms, n_neighbors, 3, n_features
#             return outputs
#         else:
#             raise ValueError(f'Unknown input dimension: {n_dim}')

#     def sum_neighbors(self, x, mask, dim:int=2, avg:bool=False):
#         """

#         Parameters
#         ----------
#         x: torch.tensor
#             usually of shape B,A,N,nf
#         mask: torch.tensor
#             usually of shape B,A,N
#         dim: int
#             the dimension to sum

#         avg: bool
#             if True, returns the average output by dividing the sum by number of neighbors.

#         Returns
#         -------

#         """
#         dim_diff = x.dim() - mask.dim()
#         for _ in range(dim_diff):
#             mask = mask.unsqueeze(-1)

#         x = x * mask
#         out = torch.sum(x, dim=dim)

#         if avg:
#             n_atoms = torch.sum(mask, dim)
#             n_atoms = torch.max(n_atoms, other=torch.ones_like(n_atoms))
#             out = out / n_atoms

#         return out

#     def forward(
#             self, 
#             invariant_node, 
#             invariant_edge, 
#             distances, 
#             distance_vector, 
#             neighbors, 
#             neighbor_mask,
#             equivariant_node_F, 
#             equivariant_node_f, 
#             equivariant_node_dr, 
#             e_dynamics,
#             ):

#         # map decomposed distances
#         invariant_edge_message = self.calculate_edge_message(invariant_edge) * self.cutoff_network(distances)[:, :, :, None]  # batch_size, n_atoms, n_neighbors, n_features

#         # map atomic features
#         invariant_node_message = self.calculate_node_message(invariant_node)  # batch_size, n_atoms, n_features

#         # copy central atom features for the element-wise multiplication
#         # ai_msij = node_message.repeat(1, 1, message.size(2))
#         # ai_msij = ai_msij.view(message.size())  # B,A,N,nf

#         # look up neighboring atoms features based on the schnet contiuous filter implementation
#         # aj_msij = self.gather_neighbors(node_message, neighbors)  # B,A,N,nf
#         # assert aj_msij is not None

#         # symmetric feature multiplication
#         # mij = message * aj_msij
#         # msij = mij * ai_msij
#         invariant_message = invariant_node_message[:, :, None, :] * self.gather_neighbors(invariant_node_message, neighbors) * invariant_edge_message  # batch_size, n_atoms, n_neighbors, n_features

#         # update a with invariance
#         if self.double_update_latent:
#             invariant_update = self.sum_neighbors(invariant_message, neighbor_mask, dim=2)    # batch_size, n_atoms, n_features
#             invariant_node = invariant_node + invariant_update    # batch_size, n_atoms, n_features

#         # Dynamics: Forces
#         equivariant_edge_message = self.calculate_equivariant_message_coefficient(invariant_message) * distance_vector  # batch_size, n_atoms, n_neighbors, 3
#         equivariant_node_message = self.calculate_equivariant_message_feature(invariant_message)    # batch_size, n_atoms, n_neighbors, n_features
#         equivariant_update_F = self.sum_neighbors(equivariant_edge_message, neighbor_mask, dim=2)    # batch_size, n_atoms, 3
#         equivariant_node_F = equivariant_node_F + equivariant_update_F    # batch_size, n_atoms, 3

#         # f
#         equivariant_message_f = equivariant_node_message[:, :, :, None, :] * equivariant_edge_message[:, :, :, :, None]    # batch_size, n_atoms, n_neighbors, 3, n_features
#         equivariant_update_f = self.sum_neighbors(equivariant_message_f, neighbor_mask, dim=2)  # batch_size, n_atoms, 3, n_features
#         equivariant_node_f = equivariant_node_f + equivariant_update_f    # batch_size, n_atoms, 3, n_features

#         # dr
#         equivariant_update_dr = self.phi_r(invariant_node)[:, :, None, :] * equivariant_update_f    # batch_size, n_atoms, 3, n_features
#         equivariant_node_dr = equivariant_node_dr + equivariant_update_dr    # batch_size, n_atoms, 3, n_features

#         equivariant_message_dr = self.phi_r_ext(invariant_message)[:, :, :, None, :] * self.gather_neighbors(equivariant_node_dr, neighbors)    # batch_size, n_atoms, n_neighbors, 3, n_features
#         equivariant_update_dr = self.sum_neighbors(equivariant_message_dr, neighbor_mask, dim=2)  # batch_size, n_atoms, 3, n_features
#         equivariant_node_dr = equivariant_node_dr + equivariant_update_dr    # batch_size, n_atoms, 3, n_features

#         # update energy
#         invariant_update = -self.phi_e(invariant_node) * torch.sum(equivariant_node_f * equivariant_node_dr, dim=-2)  # batch_size, n_atoms, n_features
#         invariant_node = invariant_node + invariant_update
#         e_dynamics = e_dynamics + invariant_update

#         # layer norm
#         if self.layer_norm:
#             invariant_node = self.norm(invariant_node)

#         return invariant_node, equivariant_node_F, equivariant_node_f, equivariant_node_dr, e_dynamics


# class AtomicEnergy(nn.Module):

#     def __init__(self, n_features, activation, dropout):
#         super(AtomicEnergy, self).__init__()
#         self.environment = nn.Sequential(
#             # Dense(n_features, 128, activation=activation, dropout=dropout, norm=False),
#             # Dense(128, 64, activation=activation, dropout=dropout, norm=False),
#             # Dense(64, 1, activation=None, dropout=0.0, norm=False),
#             nn.Linear(n_features, 128),
#             activation,
#             nn.Dropout(dropout),
#             nn.Linear(128, 64),
#             activation,
#             nn.Dropout(dropout),
#             nn.Linear(64, 1),
#         )

#     def forward(self, a):
#         # update atomic features
#         out = self.environment(a)

#         return out
