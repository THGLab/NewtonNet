import numpy as np


def euler_rotation_matrix(theta):
    """
    Rotate the xyz values based on the euler angles, theta. Directly copied from:
    Credit : "https://www.learnopencv.com/rotation-matrix-to-euler-angles/"

    Parameters
    ----------
    theta: numpy array
        A 1D array of angles along x, y and z directions

    Returns
    -------
    numpy array: rotation matrix with shape (3,3)
    """

    R_x = np.array([[1, 0, 0], [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]),
                     np.cos(theta[0])]])
    R_y = np.array([[np.cos(theta[1]), 0,
                     np.sin(theta[1])], [0, 1, 0],
                    [-np.sin(theta[1]), 0,
                     np.cos(theta[1])]])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0], [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def rotate_molecule(atoms, theta=None):
    """
    Rotates the structure of molecule between -pi/2 and pi/2.

    Parameters
    ----------
    atoms: numpy array
        An array of atomic positions with last dimension = 3

    theta: numpy array, optional (default: None)
        A 1D array of angles along x, y and z directions.
        If None, it will be generated uniformly between -pi/2 and pi/2

    Returns
    -------
    numpy array: The rotated atomic positions with shape (... , 3)

    """

    # handle theta
    if theta is None:
        theta = np.random.uniform(-np.pi / 2., np.pi / 2., size=3)

    # rotation matrix
    R = euler_rotation_matrix(theta)

    return np.dot(atoms, R)


def padaxis(array, new_size, axis, pad_value=0, pad_right=True):
    """
    Padds one axis of an array to a new size
    This is just a wrapper for np.pad, more usefull when only padding a single axis

    Parameters
    ----------
    array: ndarray
        the array to pad

    new_size: int
        the new size of the specified axis

    axis: int
        axis along which to pad

    pad_value: float or int, optional(default=0)
        pad value

    pad_right: bool, optional(default=True)
        if True pad on the right side, otherwise pad on left side

    Returns
    -------
    ndarray: padded array

    """
    add_size = new_size - array.shape[axis]
    assert add_size >= 0, 'Cannot pad dimension {0} of size {1} to smaller size {2}'.format(
        axis, array.shape[axis], new_size)
    pad_width = [(0, 0)] * len(array.shape)

    #pad after if int is provided
    if pad_right:
        pad_width[axis] = (0, add_size)
    else:
        pad_width[axis] = (add_size, 0)

    return np.pad(array,
                  pad_width=pad_width,
                  mode='constant',
                  constant_values=pad_value)

def standardize_batch(batch_content, return_mask=False):
    '''
    Helper function that pads all sequential data in a batch to be the same length

    Parameters
    ----------
    batch_content: list
        a list of contents, where each element of the list is a sequential data

    return_mask: bool
        whether or not return the mask for valid data/padding
    '''
    if type(batch_content[0]) is np.ndarray:
        all_len = [len(item) for item in batch_content]
        maximal_len = np.max(all_len)
        batch_content = np.array([padaxis(arr, maximal_len, 0) for arr in batch_content])
    else:
        assert not return_mask
        batch_content = np.array(batch_content)
    if return_mask:
        mask = np.zeros((len(batch_content), maximal_len))
        for idx, length in enumerate(all_len):
            mask[idx, :length] = 1
        return batch_content, mask
    else:
        return batch_content

def check_data_consistency(data):
    """Check that various arrays in data dictionary are consistent.

    Parameters
    ----------
    data : dict
        The dictionary containing 'R', 'Z', 'N', 'E', 'F', and 'RXN' keys and corresponding
        array values.

    """
    # check data keys
    keys = sorted(['R', 'Z', 'N', 'E', 'F', 'RXN'])
    if sorted(data.keys()) != keys and sorted(data.keys()) != sorted(['CE', 'CF', 'E', 'F', 'N', 'R', 'RXN', 'Z']):
        raise ValueError(f"Data keys {sorted(data.keys())} does not match expected {keys} key!")

    # check data arrays shape
    if data['Z'].ndim != 2:
        raise ValueError("The data['Z'] ndim {data['Z'].dim} != 2")
    n_data, n_atom = data['Z'].shape

    if data['R'].shape != (n_data, n_atom, 3):
        raise ValueError(f"The data['R'] shape {data['R'].shape} != {(n_data, n_atom, 3)}")

    if data['E'].shape != (n_data, 1):
        raise ValueError(f"The data['E'] shape {data['E'].shape} != {(n_data, 1)}")

    if data['F'].shape != (n_data, n_atom, 3):
        raise ValueError(f"The data['F'] shape {data['F'].shape} != {(n_data, n_atom, 3)}")

    if data['N'].shape != (n_data, 1):
        raise ValueError(f"The data['N'] shape {data['N'].shape} != {(n_data, 1)}")

    if data['RXN'].shape != (n_data, 1):
        raise ValueError(f"The data['RXN'] shape {data['RXN'].shape} != {(n_data, 1)}")


def combine_rxn_arrays(rxns, n_max=6):
    """Combine reaction data dictionaries into one dictionary.

    Parameters
    ----------
    rxts : list
        List of dictionaries containing various datasets.
    n_max : int
        Maximum number of atoms in reactions.

    """
    # data dictionary to store combined data
    data = {'R': None, 'Z': None, 'N': None, 'E': None, 'F': None, 'RXN': None}
    n_total = 0

    for index, rxn in enumerate(rxns):
        # check reaction keys
        if sorted(rxn.keys()) != sorted(data.keys()):
            raise ValueError(f"RXN {index} keys={sorted(rxn.keys())} != {sorted(data.keys())}")

        # check consistency of data arrays shape
        check_data_consistency(rxn)

        # get number of data points and number of atoms
        n_data, n_atom = rxn['Z'].shape

        # check n_atom
        if n_atom > n_max:
            raise ValueError(f"RXN {index} has {n_atom} atoms which is greater than n_max={n_max}")

        for key, value in rxn.items():
            # pad arrays which depend on the number of atoms
            if key in ['R', 'Z', 'F'] and n_atom < n_max:
                value = padaxis(value, n_max, 1, pad_value=0, pad_right=True)
            # concatenate data
            if index == 0:
                data[key] = np.copy(value)
            else:
                data[key] = np.concatenate([data[key], value], axis=0)
        n_total += n_data

    # check consistency of data arrays shape
    check_data_consistency(data)

    return data


def write_data_npz(data, fname):
    """Save arrays into a single compressed file using the keywords given.

    Parameters
    ----------
    data : dict
        The dictionary containing 'R', 'Z', 'N', 'E', 'F', and 'RXN' keys and array values.
    fname : str
        Filename where the data will be saved.

    """
    if not isinstance(data, dict):
        raise ValueError('Argument data is supposed to be a dictionary!')
    # check consistency of data arrays shape
    check_data_consistency(data)
    # save data
    print(f"Generate {fname} containing {data['Z'].shape[0]} data points\n")
    np.savez_compressed(fname, **data)


def get_data_subset(data, rxn_num):
    """Return data dictionary corresponding to only specified reaction number.

    Parameters
    ----------
    data : dict
        The dictionary containing 'R', 'Z', 'N', 'E', 'F', and 'RXN' keys and array values.
    rxn_num : str
        The reaction number specified as a string of length two, appended with zero, if needed.

    """
    if not isinstance(rxn_num, str) or len(rxn_num) != 2:
        raise ValueError("Argument rxn_num is expected to be a string of length 2!")

    # check consistency of data arrays shape
    check_data_consistency(data)

    # check whether rxn exists in data
    rxn_nums = np.unique(data['RXN'])
    if rxn_num not in rxn_nums:
        raise ValueError(f"Reaction {rxn_num} does not exist among existing {rxn_nums} rxns!")
    if len(rxn_nums) == 1 and rxn_nums[0] == rxn_num:
        return data

    mask = np.where(data['RXN'] == rxn_num)[0]
    sub_data = {}
    for key, value in data.items():
        if key in ['Z', 'N', 'RXN', 'E']:
            value = value[mask, :]
        elif key in ['R', 'F']:
            value = value[mask, :, :]
        sub_data[key] = value

    # check consistency of sub data arrays
    check_data_consistency(sub_data)

    return sub_data


def get_data_remove_appended_zero(data):
    """Remove appended zeros to array values of data dictionary.

    Parameters
    ----------
    data : dict
        The dictionary containing 'R', 'Z', 'N', 'E', 'F', and 'RXN' keys and array values.
        If data contains information on more than one reaction, an error is raised.

    """

    # check consistency of data arrays shape
    check_data_consistency(data)

    # get reactions numbers included in data
    rxn_nums = np.unique(data['RXN'])
    print(f"Reactions contained in data: {rxn_nums}")
    if len(rxn_nums) != 1:
        if not len(rxn_nums) ==2 and rxn_nums[0][:-1] == rxn_nums[1][:-1]: #2 spin state reaction
            raise ValueError(f"Cannot remove appended zero as {rxn_nums} rxns were found!")

    # identify unique number of atoms
    n_atom = np.unique(data['N'])
    if len(n_atom) != 1:
        raise ValueError(f"Cannot remove appended zero as {n_atom} number of atoms were found!")
    n_atom = n_atom[0]

    # identify atomic numbers
    nums = np.unique(data['Z'], axis=0)
    if len(nums) != 1:
        raise ValueError(f"Cannot remove appended zero as {nums} atomic numbers were found!")
    if np.count_nonzero(nums[0] != 0) != n_atom:
        raise ValueError(f"Cannot remove appended zero as number of non-zero atomic numbers does "
                         "not match number of atoms!")

    new_data = {}
    for key, value in data.items():
        if key == 'Z':
            value = value[:, :n_atom]
        elif key in ['R', 'F']:
            value = value[:, :n_atom, :]
        new_data[key] = value

    # check consistency of new data arrays
    check_data_consistency(new_data)

    return new_data
