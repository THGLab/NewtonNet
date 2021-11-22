import numpy as np
import torch
from torch.utils.data import Dataset

from newtonnet.utils import standardize_batch
from newtonnet.utils import rotate_molecule, euler_rotation_matrix


class BatchDataset(Dataset):
    """
    Parameters
    ----------
    input: dict
        The dictionary of batch data in ndarray format.

    """
    def __init__(self, input, device):

        self.R = torch.tensor(input['R'],
                                  device=device,
                                  # dtype=torch.float64
                              )

        self.Z = torch.tensor(input['Z'],
                                           dtype=torch.long,
                                           device=device)
        self.E = torch.tensor(input['E'],
                                   # dtype=torch.float64,
                                   device=device)
        self.F = torch.tensor(input['F'],
                                   # dtype=torch.float64,
                                   device=device)
        self.AM = torch.tensor(input['AM'],
                                dtype=torch.long,
                                device=device)
        if 'NM' in input and input['NM'] is not None:
            self.N = torch.tensor(input['N'],
                                   dtype=torch.long,
                                   device=device)
            self.NM = torch.tensor(input['NM'],
                                   dtype=torch.long,
                                   device=device)

        # rotation matrix
        self.RM = None
        if 'RM' in input and input['RM'] is not None:
            self.RM = torch.tensor(input['RM'],
                                   # dtype=torch.float64,
                                   device=device)

    def __getitem__(self, index):

        output = dict()
        output['R'] = self.R[index]
        output['Z'] = self.Z[index]
        output['E'] = self.E[index]
        output['F'] = self.F[index]
        output['AM'] = self.AM[index]
        output['N'] = self.N[index]
        output['NM'] = self.NM[index]
        if self.RM is not None:
            output['RM'] = self.RM[index]

        return output

    def __len__(self):
        return self.R.size()[0]


def batch_dataset_converter(input, device):
    result = {}
    result["R"] = torch.tensor(input['R'],
                                  # dtype=torch.float64,
                                   device=device)

    result["Z"] = torch.tensor(input['Z'],
                                           dtype=torch.long,
                                   device=device)
    if "E" in input:
        result["E"] = torch.tensor(input['E'],
                                   dtype=torch.float32,
                                   device=device)
    if "F" in input:
        result["F"] = torch.tensor(input['F'],
                                    dtype=torch.float32,
                                    device=device)
    else:
        result["F"] = None
    if "CS" in input:
        result["CS"] = torch.tensor(input['CS'],
                                    dtype=torch.float32,
                                    device=device)
    if "cs_scaler" in input:
        cs_scaler = input['cs_scaler']
        cs_scaler[cs_scaler == 0] = 1 
        result["cs_scaler"] = torch.tensor(input['cs_scaler'],
                                    dtype=torch.float32,
                                    device=device)

    if "M" in input:
        result["M"] = torch.tensor(input['M'],
                                dtype=torch.long,
                                device=device)
    result["AM"] = torch.tensor(input['AM'],
                                dtype=torch.long,
                                   device=device)
    if 'NM' in input and input['NM'] is not None:
        result["N"] = torch.tensor(input['N'],
                                   dtype=torch.long,
                                   device=device)
        result["NM"] = torch.tensor(input['NM'],
                                   dtype=torch.long,
                                   device=device)
    
    if 'lattice' in input:
        result['lattice'] = torch.tensor(input['lattice'],
                                         device=device)
    if 'D' in input:
        result['D'] = torch.tensor(input['D'],
                                    device=device)

    if 'V' in input:
        result['V'] = torch.tensor(input['V'],
                                    device=device)

    if 'NA' in input:
        result['NA'] = torch.tensor(input['NA'],
                                    dtype=torch.long,
                                    device=device)

    # rotation matrix
    result["RM"] = None
    if 'RM' in input and input['RM'] is not None:
        result["RM"] = torch.tensor(input['RM'],
                                # dtype=torch.float64,
                                   device=device)

    if "labels" in input:
        result["labels"] = input["labels"]
    return result


def extensive_train_loader(data,
                           env_provider=None,
                           batch_size=32,
                           n_rotations=0,
                           freeze_rotations=False,
                           keep_original=True,
                           device=None,
                           shuffle=True,
                           drop_last=False):
    r"""
    The main function to load and iterate data based on the extensive environment provider.

    Parameters
    ----------
    data: dict
        dictionary of arrays with following keys:
            - 'R':positions
            - 'Z':atomic_numbers
            - 'E':energy
            - 'F':forces

    env_provider: ShellProvider
        the instance of combust.data.ExtensiveEnvironment or combust.data.PeriodicEnvironment

    batch_size: int, optional (default: 32)
        The size of output tensors

    n_rotations: int, optional (default: 0)
        Number of times to rotate voxel boxes for data augmentation.
        If zero, the original orientation will be used.

    freeze_rotations: bool, optional (default: False)
        If True rotation angles will be determined and fixed during generation.

    keep_original: bool, optional (default: True)
        If True, the original orientation of data is kept in each epoch

    device: torch.device
        either cpu or gpu (cuda) device.

    shuffle: bool, optional (default: True)
        If ``True``, shuffle the list of file path and batch indices between iterations.

    drop_last: bool, optional (default: False)
        set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (default: ``False``)

    Yields
    -------
    BatchDataset: instance of BatchDataset with the all batch data

    """
    n_data = data['R'].shape[0]  # D
    # n_atoms = data['R'].shape[1]  # A

    # print("number of data w/o augmentation: ", n_data)



    # # shuffle
    # if shuffle:
    #     shuffle_idx = np.arange(n_data)
    #     carts = carts[shuffle_idx]
    #     atomic_numbers = atomic_numbers[shuffle_idx]
    #     energy = energy[shuffle_idx]
    #     forces = forces[shuffle_idx]

    if isinstance(freeze_rotations, list):
        thetas = freeze_rotations
        freeze_rotations = True
    else:
        if freeze_rotations:
            import itertools
            theta_gen = itertools.permutations(np.linspace(-np.pi, np.pi, 6), 3) #todo: linspace size
            thetas = [np.array([0.0, 0.0, 0.0])
                      ]  # index 0 is reserved for the original data (no rotation)
            thetas += [
                np.random.uniform(-np.pi, np.pi, size=3)
                for _ in range(n_rotations)
            ]

    # iterate over data snapshots
    seen_all_data = 0
    while True:

        # iterate over rotations
        for r in range(n_rotations + 1):

            # split by batch size and yield
            data_atom_indices = list(range(n_data))

            if shuffle:
                np.random.shuffle(data_atom_indices)

            split = 0
            while (split + 1) * batch_size <= n_data:
                # Output a batch
                data_batch_idx = data_atom_indices[split *
                                                   batch_size:(split + 1) *
                                                   batch_size]
                data_batch = {k: standardize_batch(data[k][data_batch_idx]) for k in data if k != "labels"}
                if "labels" in data:
                    data_batch["labels"] = [data["labels"][idx] for idx in data_batch_idx]
                # get neighbors
                if env_provider is not None:
                    if 'lattice' in data_batch:
                        neighbors, neighbor_mask, atom_mask, distances, distance_vectors = \
                        env_provider.get_environment(data_batch['R'], data_batch['Z'], data_batch['lattice'])
                    else:
                        neighbors, neighbor_mask, atom_mask, distances, distance_vectors = \
                        env_provider.get_environment(data_batch['R'], data_batch['Z'])

                # rotation
                if r > 0:
                    if freeze_rotations:
                        try:
                            theta = np.array(next(theta_gen))
                        except:
                            theta_gen = itertools.permutations(
                            np.linspace(-np.pi, np.pi, 6), 3)
                            theta = np.array(next(theta_gen))
                    else:
                        theta = np.random.uniform(-np.pi, np.pi, size=3)
                    rot_atoms = rotate_molecule(data_batch['R'], theta=theta)  # B, A, 3
                    if distance_vectors is not None:
                        rot_dist_vec = rotate_molecule(distance_vectors, theta=theta)
                    if 'F' in data:
                        rot_forces = rotate_molecule(data_batch['F'], theta=theta)  # B, A, 3
                else:
                    if not keep_original:
                        theta = np.random.uniform(-np.pi, np.pi, size=3)
                        rot_atoms = rotate_molecule(data_batch['R'], theta=theta)  # B, A, 3
                        if distance_vectors is not None:
                            rot_dist_vec = rotate_molecule(distance_vectors, theta=theta)
                        if 'F' in data:
                            rot_forces = rotate_molecule(data_batch['F'], theta=theta)  # B, A, 3
                    else:
                        theta = np.array([0,0,0])
                        rot_atoms = data_batch['R']
                        if 'F' in data:
                            rot_forces = data_batch['F']
                        if distance_vectors is not None:
                            rot_dist_vec = distance_vectors

                RM = np.tile(theta, (rot_atoms.shape[0],1))

                if env_provider is None:
                    N = None
                    NM = None
                    Z = data_batch['Z']
                    AM = np.zeros_like(Z)
                    AM[Z != 0] = 1
                else:
                    N = neighbors
                    NM = neighbor_mask
                    AM = atom_mask

                batch_dataset = {k:v for k,v in data_batch.items()}
                batch_dataset.update({'R': rot_atoms,  # B,A,3
                                      'N': N,     # B,A,A-1
                                      'NM': NM,   # B,A,A-1
                                      'AM': AM,   # B,A
                                      'RM': RM    # B,3  rotation angles only
                                      })
                if 'F' in data:
                    batch_dataset.update({'F': rot_forces})

                if distances is not None and distance_vectors is not None:
                    batch_dataset.update({'D': distances,
                                          'V': rot_dist_vec})

                # batch_dataset = {
                #     'R': rot_atoms,   # B,A,3
                #     'Z': data_batch['Z'], # B,A
                #     'E': data_batch['E'], # B,1
                #     'F': rot_forces if 'F' in data else None,    # B,A,3
                #     'N': N,     # B,A,A-1
                #     'NM': NM,   # B,A,A-1
                #     'AM': AM,   # B,A
                #     'RM': RM    # B,3  rotation angles only
                # }
                # batch_dataset = BatchDataset(batch_dataset, device=device)
                batch_dataset = batch_dataset_converter(batch_dataset, device)
                yield batch_dataset
                split += 1

            # Deal with the part smaller than a batch_size
            left_len = n_data % batch_size
            if left_len != 0 and drop_last:
                continue

            elif left_len != 0 and not drop_last:
                left_idx = data_atom_indices[split * batch_size:]
                data_batch = {k: standardize_batch(data[k][left_idx]) for k in data for k in data if k != "labels"}
                if "labels" in data:
                    data_batch["labels"] = [data["labels"][idx] for idx in left_idx]
                # get neighbors
                if env_provider is not None:
                    if 'lattice' in data:
                        neighbors, neighbor_mask, atom_mask, distances, distance_vectors = \
                        env_provider.get_environment(data_batch['R'], data_batch['Z'], data_batch['lattice'])
                    else:
                        neighbors, neighbor_mask, atom_mask, distances, distance_vectors = \
                        env_provider.get_environment(data_batch['R'], data_batch['Z'])

                # rotation
                if r > 0:
                    if freeze_rotations:
                        try:
                            theta = np.array(next(theta_gen))
                        except:
                            theta_gen = itertools.permutations(
                            np.linspace(-np.pi, np.pi, 6), 3)
                            theta = np.array(next(theta_gen))
                    else:
                        theta = np.random.uniform(-np.pi, np.pi, size=3)
                    rot_atoms = rotate_molecule(data_batch['R'], theta=theta)  # B, A, 3
                    if distance_vectors is not None:
                        rot_dist_vec = rotate_molecule(distance_vectors, theta=theta)
                    if 'F' in data:
                        rot_forces = rotate_molecule(data_batch['F'], theta=theta)  # B, A, 3
                else:
                    if not keep_original:
                        theta = np.random.uniform(-np.pi, np.pi, size=3)
                        rot_atoms = rotate_molecule(data_batch['R'], theta=theta)  # B, A, 3
                        if distance_vectors is not None:
                            rot_dist_vec = rotate_molecule(distance_vectors, theta=theta)
                        if 'F' in data:
                            rot_forces = rotate_molecule(data_batch['F'], theta=theta)  # B, A, 3
                    else:
                        theta = np.array([0,0,0])
                        rot_atoms = data_batch['R']
                        if 'F' in data:
                            rot_forces = data_batch['F']
                        if distance_vectors is not None:
                            rot_dist_vec = distance_vectors

                RM = np.tile(theta, (rot_atoms.shape[0],1))

                if env_provider is None:
                    N = None
                    NM = None
                    Z = data_batch['Z']
                    AM = np.zeros_like(Z)
                    AM[Z != 0] = 1
                else:
                    N = neighbors
                    NM = neighbor_mask
                    AM = atom_mask

                batch_dataset = {k:v for k,v in data_batch.items()}
                batch_dataset.update({'R': rot_atoms,  # B,A,3
                                      'N': N,     # B,A,A-1
                                      'NM': NM,   # B,A,A-1
                                      'AM': AM,   # B,A
                                      'RM': RM    # B,3  rotation angles only
                                      })

                if 'F' in data:
                    batch_dataset.update({'F': rot_forces})

                if distances is not None and distance_vectors is not None:
                    batch_dataset.update({'D': distances,
                                          'V': rot_dist_vec})

                # batch_dataset = {
                #     'R': rot_atoms,
                #     'Z': data_batch['Z'],
                #     'E': data_batch['E'],
                #     'F': rot_forces if 'F' in data else None,
                #     'N': N,
                #     'NM': NM,
                #     'AM': AM,
                #     'RM': RM
                # }

                # batch_dataset = BatchDataset(batch_dataset, device)
                batch_dataset = batch_dataset_converter(batch_dataset, device)
                yield batch_dataset

            seen_all_data += 1
            # print('\n# trained on entire data: %i (# rotation: %i)\n'%(seen_all_data, (n_rotations+1)))


def extensive_loader_rotwise(
                           data,
                           env_provider=None,
                           batch_size=32,
                           n_rotations=0,
                           freeze_rotations=False,
                           keep_original=True,
                           device=None,
                           shuffle=True,
                           drop_last=False):
    r"""
    The main function to load and iterate data based on the extensive environment provider.
    In this implementation we keep track of all rotations of a data point (compared to `extensive_train_loader`)

    Parameters
    ----------
    data: dict
        dictionary of arrays with following keys:
            - 'R':positions
            - 'Z':atomic_numbers
            - 'E':energy
            - 'F':forces

    env_provider: ShellProvider
        the instance of combust.data.ExtensiveEnvironment

    batch_size: int, optional (default: 32)
        The size of output tensors

    n_rotations: int, optional (default: 0)
        Number of times to rotate voxel boxes for data augmentation.
        If zero, the original orientation will be used.

    freeze_rotations: bool, optional (default: False)
        If True rotation angles will be determined and fixed during generation.

    keep_original: bool, optional (default: True)
        If True, the original orientation of data is kept in each epoch

    device: torch.device
        either cpu or gpu (cuda) device.

    shuffle: bool, optional (default: True)
        If ``True``, shuffle the list of file path and batch indices between iterations.

    drop_last: bool, optional (default: False)
        set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (default: ``False``)

    Yields
    -------
    BatchDataset: instance of BatchDataset with the all batch data

    """
    n_data = data['R'].shape[0]  # D
    n_atoms = data['R'].shape[1]  # A

    # print("number of data w/o augmentation: ", n_data)

    # get neighbors
    if env_provider is not None:
        neighbors, neighbor_mask, atom_mask = env_provider.get_environment(data['R'], data['Z'])

    # # shuffle
    # if shuffle:
    #     shuffle_idx = np.arange(n_data)
    #     carts = carts[shuffle_idx]
    #     atomic_numbers = atomic_numbers[shuffle_idx]
    #     energy = energy[shuffle_idx]
    #     forces = forces[shuffle_idx]

    if isinstance(freeze_rotations, list):
        thetas = freeze_rotations
        freeze_rotations = True
    else:
        if freeze_rotations:
            thetas = [np.array([0.0, 0.0, 0.0])
                      ]  # index 0 is reserved for the original data (no rotation)
            thetas += [
                np.random.uniform(-np.pi, np.pi, size=3)
                for _ in range(n_rotations)
            ]

    # iterate over data snapshots
    seen_all_data = 0
    while True:

        # split by batch size and yield
        data_atom_indices = list(range(n_data))

        if shuffle:
            np.random.shuffle(data_atom_indices)

        split = 0
        while (split + 1) * batch_size <= n_data:
            # Output a batch
            data_batch_idx = data_atom_indices[split *
                                               batch_size:(split + 1) *
                                               batch_size]

            # rotation
            if not freeze_rotations:
                if keep_original:
                    thetas = np.random.uniform(-np.pi, np.pi, size=(n_rotations, 3))
                    thetas = [np.array([0,0,0])] + list(thetas)
                else:
                    thetas = np.random.uniform(-np.pi, np.pi, size=(n_rotations+1, 3))

            # stack all rotations
            Rs=[]; Fs=[]; RMs=[]
            for theta in thetas:
                rot_atoms = rotate_molecule(data['R'][data_batch_idx], theta=theta)  # B, A, 3
                # rot_forces = rotate_molecule(data['F'][data_batch_idx], theta=theta)  # B, A, 3
                rot_matrix = euler_rotation_matrix(theta)   # 3,3

                Rs.append(rot_atoms)
                # Fs.append(rot_forces)
                RMs.append(rot_matrix)

            rot_atoms = np.stack(Rs, axis=1)  # B,n_rot,A,3
            rsize = rot_atoms.shape
            rot_atoms = rot_atoms.reshape(rsize[0]*rsize[1], rsize[2], rsize[3]) # B*n_rot, A, 3
            # rot_forces= np.stack(Fs, axis=1)  # B,n_rot,A,3
            # rot_forces= rot_forces.reshape(rsize[0]*rsize[1], rsize[2], rsize[3]) # B*n_rot, A, 3

            RM = np.stack(RMs, axis=0)  # n_rot,3,3
            RM = np.tile(RM, (rsize[0], 1, 1, 1))    # B, n_rot, 3, 3

            if env_provider is None:
                N = None
                NM = None
                E = data['E'][data_batch_idx]  # B,1
                # E = np.tile(E, (1, n_rotations + 1)).reshape(E.shape[0] * (n_rotations + 1), E.shape[1])  # B*n_rot,1

                Z = data['Z'][data_batch_idx]  # B,A
                Z = np.tile(Z, (1,n_rotations+1)).reshape(Z.shape[0],n_rotations+1,Z.shape[1]) # B,n_rot,A

                AM = np.zeros_like(Z)   # B,n_rot,A
                AM[Z != 0] = 1
                RM = RM

            else:
                E = data['E'][data_batch_idx]  # B,1
                # E = np.tile(E, (1, n_rotations + 1)).reshape(E.shape[0] * (n_rotations + 1), E.shape[1])  # B*n_rot,1

                Z = data['Z'][data_batch_idx]  # B,A
                Z = np.tile(Z, (1,n_rotations+1)).reshape(Z.shape[0]*(n_rotations+1),Z.shape[1]) # B*n_rot,A

                N = neighbors[data_batch_idx]  # B,A,A-1
                N = np.tile(N, (1,n_rotations+1,1)).reshape(N.shape[0]*(n_rotations+1),N.shape[1], N.shape[2]) # B*n_rot,A,A-1

                NM = neighbor_mask[data_batch_idx]  # B,A,A-1
                NM = np.tile(NM, (1,n_rotations+1,1)).reshape(NM.shape[0]*(n_rotations+1),NM.shape[1], NM.shape[2]) # B*n_rot,A,A-1

                AM = atom_mask[data_batch_idx]   # B,A
                AM = np.tile(AM, (1,n_rotations+1)).reshape(AM.shape[0]*(n_rotations+1),AM.shape[1]) # B*n_rot,A

                # rotation mask
                # B = rsize[0]; n_rot=rsize[1]
                # RotM = np.eye(B)
                # RotM = np.tile(RotM, (1, n_rot)).reshape(B*n_rot, B)

                RM = RM

            batch_dataset = {
                'R': rot_atoms,   # B,A,3
                'Z': Z, # B,A
                'E': E, # B,1
                'F': data['F'][data_batch_idx],    # B,A,3
                'N': N,     # B,A,A-1
                'NM': NM,   # B,A,A-1
                'AM': AM,   # B,A
                'RM': RM
            }

            # batch_dataset = BatchDataset(batch_dataset, device=device)
            batch_dataset = batch_dataset_converter(batch_dataset, device)
            yield batch_dataset
            split += 1

        # Deal with the part smaller than a batch_size
        left_len = n_data % batch_size
        if left_len != 0 and drop_last:
            continue

        elif left_len != 0 and not drop_last:
            left_idx = data_atom_indices[split * batch_size:]

            # rotation
            if not freeze_rotations:
                if keep_original:
                    thetas = np.random.uniform(-np.pi, np.pi, size=(n_rotations, 3))
                    thetas = [np.array([0,0,0])] + list(thetas)
                else:
                    thetas = np.random.uniform(-np.pi, np.pi, size=(n_rotations+1, 3))

            # stack all rotations
            Rs=[]; Fs=[]; RMs=[]
            for theta in thetas:
                rot_atoms = rotate_molecule(data['R'][left_idx], theta=theta)  # B, A, 3
                # rot_forces = rotate_molecule(data['F'][left_idx], theta=theta)  # B, A, 3
                rot_matrix = euler_rotation_matrix(theta)   # 3,3

                Rs.append(rot_atoms)
                # Fs.append(rot_forces)
                RMs.append(rot_matrix)

            rot_atoms = np.stack(Rs, axis=1)  # B,n_rot,A,3
            rsize = rot_atoms.shape
            rot_atoms = rot_atoms.reshape(rsize[0]*rsize[1], rsize[2], rsize[3]) # B*n_rot, A, 3
            # rot_forces= np.stack(Fs, axis=1)  # B,n_rot,A,3
            # rot_forces= rot_forces.reshape(rsize[0]*rsize[1], rsize[2], rsize[3]) # B*n_rot, A, 3

            RM = np.stack(RMs, axis=0)  # n_rot,3,3
            RM = np.tile(RM, (rsize[0], 1, 1, 1))    # B, n_rot, 3, 3

            if env_provider is None:
                N = None
                NM = None

                E = data['E'][left_idx]  # B,1
                # E = np.tile(E, (1,n_rotations+1)).reshape(E.shape[0]*(n_rotations+1),E.shape[1]) # B*n_rot,1

                Z = data['Z'][left_idx]
                Z = np.tile(Z, (1,n_rotations+1)).reshape(Z.shape[0]*(n_rotations+1),Z.shape[1]) # B*n_rot,A

                AM = np.zeros_like(Z)
                AM[Z != 0] = 1
                RM = RM
            else:
                E = data['E'][left_idx]  # B,1
                # E = np.tile(E, (1,n_rotations+1)).reshape(E.shape[0]*(n_rotations+1),E.shape[1]) # B*n_rot,1

                Z = data['Z'][left_idx]  # B,A
                Z = np.tile(Z, (1,n_rotations+1)).reshape(Z.shape[0]*(n_rotations+1),Z.shape[1]) # B*n_rot,A

                N = neighbors[left_idx]  # B,A,A-1
                N = np.tile(N, (1,n_rotations+1,1)).reshape(N.shape[0]*(n_rotations+1),N.shape[1], N.shape[2]) # B*n_rot,A,A-1

                NM = neighbor_mask[left_idx]  # B,A,A-1
                NM = np.tile(NM, (1,n_rotations+1,1)).reshape(NM.shape[0]*(n_rotations+1),NM.shape[1], NM.shape[2]) # B*n_rot,A,A-1

                AM = atom_mask[left_idx]   # B,A
                AM = np.tile(AM, (1,n_rotations+1)).reshape(AM.shape[0]*(n_rotations+1),AM.shape[1]) # B*n_rot,A

                RM = RM

            batch_dataset = {
                'R': rot_atoms,
                'Z': Z,
                'E': E,
                'F': data['F'][left_idx],
                'N': N,
                'NM': NM,
                'AM': AM,
                'RM': RM
            }

            # batch_dataset = BatchDataset(batch_dataset, device)
            batch_dataset = batch_dataset_converter(batch_dataset, device)
            yield batch_dataset

        seen_all_data += 1
        # print('\n# trained on entire data: %i (# rotation: %i)\n'%(seen_all_data, (n_rotations+1)))

def extensive_voxel_loader(data,
                           env_provider=None,
                           batch_size=32,
                           device=None,
                           shuffle=True,
                           drop_last=False):
    r"""
    The main function to load and iterate data based on the extensive environment provider.

    Parameters
    ----------
    data: dict
        dictionary of arrays with following keys:
            - 'R':positions
            - 'Z':atomic_numbers
            - 'E':energy
            - 'F':forces

    env_provider: ShellProvider
        the instance of combust.data.ExtensiveEnvironment

    batch_size: int, optional (default: 32)
        The size of output tensors

    device: torch.device
        either cpu or gpu (cuda) device.

    shuffle: bool, optional (default: True)
        If ``True``, shuffle the list of file path and batch indices between iterations.

    drop_last: bool, optional (default: False)
        set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (default: ``False``)

    Yields
    -------
    BatchDataset: instance of BatchDataset with the all batch data

    """
    n_data = data['R'].shape[0]  # D
    n_atoms = data['R'].shape[1]  # A

    # print("number of data w/o augmentation: ", n_data)

    # get neighbors
    if env_provider is not None:
        neighbors, neighbor_mask, atom_mask = env_provider.get_environment(data['R'], data['Z'])

    # iterate over data snapshots
    seen_all_data = 0
    while True:
        # split by batch size and yield
        data_atom_indices = list(range(n_data))

        if shuffle:
            np.random.shuffle(data_atom_indices)

        split = 0
        while (split + 1) * batch_size <= n_data:
            # Output a batch
            data_batch_idx = data_atom_indices[split *
                                               batch_size:(split + 1) *
                                               batch_size]

            if env_provider is None:
                N = None
                NM = None
                Z = data['Z'][data_batch_idx]
                AM = np.zeros_like(Z)
                AM[Z != 0] = 1
                thetas = np.random.uniform(-np.pi, np.pi, size=(Z.shape[0],3))
                RM = np.array([euler_rotation_matrix(t) for t in thetas])
            else:
                N = neighbors[data_batch_idx]
                NM = neighbor_mask[data_batch_idx]
                AM = atom_mask[data_batch_idx]
                RM = None


            batch_dataset = {
                'R': data['R'][data_batch_idx],
                'Z': data['Z'][data_batch_idx],
                'E': data['E'][data_batch_idx],
                'F': data['F'][data_batch_idx],
                'N': N,
                'NM': NM,
                'AM': AM,
                'RM': RM
            }
            # batch_dataset = BatchDataset(batch_dataset, device=device)
            batch_dataset = batch_dataset_converter(batch_dataset, device)
            yield batch_dataset
            split += 1

        # Deal with the part smaller than a batch_size
        left_len = n_data % batch_size
        if left_len != 0 and drop_last:
            continue

        elif left_len != 0 and not drop_last:

            left_idx = data_atom_indices[split * batch_size:]

            if env_provider is None:
                N = None
                NM = None
                Z = data['Z'][left_idx]
                AM = np.zeros_like(Z)
                AM[Z != 0] = 1
                thetas = np.random.uniform(-np.pi, np.pi, size=(Z.shape[0],3))
                RM = np.array([euler_rotation_matrix(t) for t in thetas])
            else:
                N = neighbors[left_idx]
                NM = neighbor_mask[left_idx]
                AM = atom_mask[left_idx]
                RM = None

            batch_dataset = {
                'R': data['R'][left_idx],
                'Z': data['Z'][left_idx],
                'E': data['E'][left_idx],
                'F': data['F'][left_idx],
                'N': N,
                'NM': NM,
                'AM': AM,
                'RM': RM
            }

            # batch_dataset = BatchDataset(batch_dataset, device)
            batch_dataset = batch_dataset_converter(batch_dataset, device)
            yield batch_dataset

            seen_all_data += 1
            # print('\n# trained on entire data: %i (# rotation: %i)\n'%(seen_all_data, (n_rotations+1)))


