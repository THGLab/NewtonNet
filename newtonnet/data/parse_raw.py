import os
import numpy as np
import warnings
from collections import defaultdict
from numpy.lib.function_base import append
from sklearn.utils import random
from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import train_test_split

from newtonnet.data import ExtensiveEnvironment, PeriodicEnvironment
from newtonnet.data import extensive_train_loader, extensive_loader_rotwise

from ase.io import iread
import math
import pickle

def concat_listofdicts(listofdicts, axis=0):
    """

    Parameters
    ----------
    listofdicts: list
        values must be 2d arrays
    axis: int

    Returns
    -------
    dict

    """
    data = dict()
    for k in listofdicts[0].keys():
        data[k] = np.concatenate([d[k] for d in listofdicts], axis=axis)

    return data


def split(data, train_size, test_size, val_size, random_states=90, stratify=None):
    """

    Parameters
    ----------
    data: dict
    train_size: int
    test_size
    val_size
    random_states
    stratify: None or labels

    Returns
    -------
    dict: train data
    dict: val data
    dict: test data

    """

    tr_ind, val_ind = train_test_split(list(range(data['R'].shape[0])),
                                      test_size=val_size,
                                      random_state=random_states,
                                      stratify=stratify)

    if stratify is not None:
        stratify_new = stratify[tr_ind]
    else:
        stratify_new = None

    tr_ind, te_ind = train_test_split(tr_ind,
                                       test_size=test_size,
                                       train_size=train_size,
                                       random_state=random_states,
                                       stratify=stratify_new)

    train = dict()
    val = dict()
    test = dict()
    for key in data:
        train[key] = data[key][tr_ind]
        val[key] = data[key][val_ind]
        test[key] = data[key][te_ind]

    if stratify is not None:
        train['L'] = stratify[tr_ind]
        val['L'] = stratify[val_ind]
        test['L'] = stratify[te_ind]

    return train, val, test


def h2_reaction(reaction_number, settings, all_sets):
    """

    Parameters
    ----------
    reaction_number: int

    settings: dict
        dict of yaml file

    all_sets: dict

    Returns
    -------

    """

    dir_path = settings['data']['root']

    # file name prefix
    if reaction_number < 10:
        pre = '0%i'%reaction_number
    elif reaction_number >= 10:
        pre = '%i'%reaction_number
    # elif reaction_number == 6:
    #     pre = ['0%ia_irc.npz' % reaction_number, '0%ib_irc.npz' % reaction_number]
    # elif reaction_number == 12:
    #     pre = ['%ia_irc.npz' % reaction_number, '%ib_irc.npz' % reaction_number]

    # read npz files
    aimd = nm = irc = None
    aimd_path = os.path.join(dir_path, '%s_aimd.npz'%pre)
    if os.path.exists(aimd_path):
        aimd = dict(np.load(aimd_path))
    nm_path = os.path.join(dir_path, '%s_nm.npz'%pre)
    if os.path.exists(nm_path):
        nm = dict(np.load(nm_path))
    irc_path = os.path.join(dir_path, '%s_irc.npz'%pre)
    if os.path.exists(irc_path):
        irc = dict(np.load(irc_path))

    # merge aimd and normal mode data
    if settings['data']['normal_mode'] and nm is not None:
        data = dict()
        n_nm = min(settings['data']['size_nmode_max'], nm['R'].shape[0])
        nm_select = sample_without_replacement(nm['R'].shape[0],
                                               n_nm,
                                               random_state=settings['data']['random_states'])
        if aimd is not None:
            for k in aimd.keys():
                data[k] = np.concatenate([aimd[k], nm[k][nm_select]], axis=0)

            assert data['R'].shape[0] == (aimd['R'].shape[0]+n_nm)
        else:
            data = None
            warnings.warn('both AIMD and normal mode data for reaction# %i are missing.'%reaction_number)

    elif aimd is not None:
        data = aimd

    else:
        data = None
        warnings.warn('both AIMD and normal mode data for reaction# %i are missing.'%reaction_number)

    if settings['data']['cgem']:
        assert data['E'].shape == data['CE'].shape
        assert data['F'].shape == data['CF'].shape
        data['E'] = data['E'] - data['CE']
        data['F'] = data['F'] - data['CF']
        irc['E'] = irc['E'] - irc['CE']
        irc['F'] = irc['F'] - irc['CF']

    train_size = settings['data']['trsize_perrxn_max']
    if train_size == -1:
        train_size = None  # to select all remaining data in each reaction

    if data is not None:
        dtrain, dval, dtest = split(data,
                                    train_size=train_size,
                                    test_size=settings['data']['test_size'],
                                    val_size=settings['data']['val_size'],
                                    random_states=settings['data']['random_states'],
                                    stratify=None)
    else:
        dtrain, dval, dtest = None, None, None

    # compile data sets
    all_sets['train'].append(dtrain)
    all_sets['val'].append(dval)
    all_sets['test'].append(dtest)
    all_sets['irc'].append(irc)

    return all_sets


def parse_h2_reaction(settings, device):
    """

    Parameters
    ----------
    settings: instance of yaml file
    device: torch devices

    Returns
    -------
    generator: train, val, irc, test generators, respectively
    int: n_steps for train, val, irc, test, respectively
    tuple: tuple of mean and standard deviation of energies in the training data

    """

    # list of reaction number(s)
    reaction_number = settings['data']['reaction']

    if isinstance(reaction_number, int):
        reaction_number = [reaction_number]

    # compile dictionary of train, test, val, and irc data
    all_sets = defaultdict(list)
    for rxn_n in reaction_number:
        all_sets = h2_reaction(rxn_n, settings, all_sets)

    dtrain = concat_listofdicts(all_sets['train'], axis=0)
    dval = concat_listofdicts(all_sets['val'], axis=0)
    dtest = concat_listofdicts(all_sets['test'], axis=0)
    irc = concat_listofdicts(all_sets['irc'], axis=0)

    # final down-sampling of training data
    n_train = settings['data']['train_size']
    if n_train == -1:
        n_train = dtrain['R'].shape[0]

    n_train = min(n_train, dtrain['R'].shape[0])
    n_select = sample_without_replacement(dtrain['R'].shape[0],
                                           n_train,
                                           random_state=settings['data']['random_states'])
    for k in dtrain.keys():
        dtrain[k] = dtrain[k][n_select]

    normalizer = (dtrain['E'].mean(), dtrain['E'].std())

    n_tr_data = dtrain['R'].shape[0]
    n_val_data = dval['R'].shape[0]
    n_irc_data = irc['R'].shape[0]
    n_test_data = dtest['R'].shape[0]
    print("# data (train,val,test,irc): %i, %i, %i, %i"%(n_tr_data,n_val_data,n_test_data,n_irc_data))

    tr_batch_size = settings['training']['tr_batch_size']
    val_batch_size = settings['training']['val_batch_size']
    tr_rotations = settings['training']['tr_rotations']
    val_rotations = settings['training']['val_rotations']

    # freeze rotatios
    # Todo: it seems that we don't need separated tr and val anymore
    # Todo: consider keep_original scenario in the code
    # if settings['training']['tr_frz_rot']:
    #     if settings['training']['saved_angle_path']:
    #         tr_fra_rot = list(np.load(settings['training']['saved_angle_path']))[:tr_rotations+1]
    #     tr_frz_rot = (np.random.uniform(-np.pi, np.pi, size=3)
    #                   for _ in range(tr_rotations+1))
    #     val_frz_rot = tr_frz_rot
    # else:
    #     tr_frz_rot = settings['training']['tr_frz_rot']
    #     val_frz_rot = settings['training']['val_frz_rot']

    # generators
    project = settings['general']['driver']
    if project not in ['voxel_cart_rotwise.py']:
        # steps
        tr_steps = int(np.ceil(n_tr_data / tr_batch_size)) * (tr_rotations + 1)
        val_steps = int(np.ceil(n_val_data / val_batch_size)) * (val_rotations + 1)
        irc_steps = int(np.ceil(n_irc_data / val_batch_size)) * (val_rotations + 1)
        test_steps= int(np.ceil(n_test_data / val_batch_size)) * (val_rotations + 1)

        env = ExtensiveEnvironment()

        train_gen = extensive_train_loader(data=dtrain,
                                           env_provider=env,
                                           batch_size=tr_batch_size,
                                           n_rotations=tr_rotations,
                                           freeze_rotations=settings['training']['tr_frz_rot'],
                                           keep_original=settings['training']['tr_keep_original'],
                                           device=device,
                                           shuffle=settings['training']['shuffle'],
                                           drop_last=settings['training']['drop_last'])

        val_gen = extensive_train_loader(data=dval,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=settings['training']['shuffle'],
                                         drop_last=settings['training']['drop_last'])

        irc_gen = extensive_train_loader(data=irc,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=False,
                                         drop_last=False)

        test_gen = extensive_train_loader(data=dtest,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=False,
                                         drop_last=False)

        return train_gen, val_gen, irc_gen, test_gen, tr_steps, val_steps, irc_steps, test_steps, normalizer

    else:

        tr_steps = int(np.ceil(n_tr_data / tr_batch_size))
        val_steps = int(np.ceil(n_val_data / val_batch_size))
        irc_steps = int(np.ceil(n_irc_data / val_batch_size))
        test_steps = int(np.ceil(n_test_data / val_batch_size))

        env = ExtensiveEnvironment()

        train_gen = extensive_loader_rotwise(data=dtrain,
                                           env_provider=env,
                                           batch_size=tr_batch_size,
                                           n_rotations=tr_rotations,
                                           freeze_rotations=settings['training']['tr_frz_rot'],
                                           keep_original=settings['training']['tr_keep_original'],
                                           device=device,
                                           shuffle=settings['training']['shuffle'],
                                           drop_last=settings['training']['drop_last'])

        val_gen = extensive_loader_rotwise(data=dval,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=settings['training']['shuffle'],
                                         drop_last=settings['training']['drop_last'])

        irc_gen = extensive_loader_rotwise(data=irc,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=False,
                                         drop_last=False)

        test_gen = extensive_loader_rotwise(data=dtest,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=False,
                                         drop_last=False)

        return train_gen, val_gen, irc_gen, test_gen, tr_steps, val_steps, irc_steps, test_steps, normalizer


def parse_nmr_data(settings, device, test_only=False):
    '''
    parse and load NMR data from tripeptides/3d fragmentations of SPARTA+/SHIFTX2 dataset with NMR chemical shifts calculated with HF/6-31g* or experimental chemical shifts

    Parameters
    ----------
    settings: instance of yaml file
    device: list
        list of torch devices
    test_only: boolean indicator for whether only read test data

    generator: train, val, test generators, respectively
    int: n_steps for train, val, test, respectively
    tuple: tuple of mean and standard deviation of energies in the training data
    '''
    # meta data
    root_folder = settings['data']['root']

    test_proportion = settings['data']['test_proportion']
    val_proportion = settings['data']['val_proportion']
    shift_types = settings['data']['shift_types']

    dtrain = {'R':[], 'Z':[], 'N':[], 'CS':[], "M": []}
    dtest = {'R':[], 'Z':[], 'N':[], 'CS':[], "M": [], "labels": []}

    pbc = settings["data"].get("pbc", False)
    if pbc:
        dtrain["lattice"] = []
        dtest["lattice"] = []

    if test_only:
        test_path = settings['data']['test']
        test_proteins = os.listdir(test_path)
    else:
        if settings['data']['test']:
            test_path = settings['data']['test']
            train_proteins = os.listdir(root_folder)
            test_proteins = os.listdir(test_path)
        else:
            test_path = root_folder
            train_proteins, test_proteins = train_test_split(os.listdir(root_folder),
                        test_size=test_proportion,
                        random_state=settings['data']['random_states'])
        # Load train data
        for protein in train_proteins:
            with open(os.path.join(root_folder, protein), "rb") as f:
                data = pickle.load(f)
            for residue in data:
                if residue['R'].shape[1] > settings['data']['max_natom_cutoff']:
                    continue
                dtrain['R'].append(residue['R'])
                dtrain['Z'].append(residue['Z'])
                dtrain['N'].append([len(residue['R'])])
                shift_type_filter = np.array([item in shift_types for item in residue['Z']])
                dtrain['CS'].append(np.nan_to_num(residue['CS']))
                dtrain['M'].append(residue['M'] & shift_type_filter)
                if pbc:
                    dtrain['lattice'].append(residue['lattice'])


    # Load test data
    for protein in test_proteins:
        with open(os.path.join(test_path, protein), "rb") as f:
            data = pickle.load(f)
        for residue in data:
            if residue['R'].shape[0] > settings['data']['max_natom_cutoff']:
                continue
            dtest['R'].append(residue['R'])
            dtest['Z'].append(residue['Z'])
            dtest['N'].append([len(residue['R'])])
            shift_type_filter = np.array([item in shift_types for item in residue['Z']])
            dtest['CS'].append(np.nan_to_num(residue['CS']))
            dtest['M'].append(residue['M'] & shift_type_filter)
            dtest["labels"].append(residue["meta"])
            if pbc:
                dtest['lattice'].append(residue['lattice'])

    if test_only:
        n_tr_data = n_val_data = 0
        n_test_data = len(dtest['R'])
        for k in dtest:
            dtest[k] = np.array(dtest[k])
        print("test data size: %d"%n_test_data)
    else:
        # Further split train into train/validation
        train_val_indices = list(range(len(dtrain['R'])))
        train_idx, val_idx = train_test_split(train_val_indices, test_size=val_proportion, random_state=settings['data']['random_states'])
        data = dtrain
        dtrain = {}
        dval = {}
        for k in data:
            dtrain[k] = np.array(data[k])[train_idx]
            dval[k] = np.array(data[k])[val_idx]
            dtest[k] = np.array(dtest[k])

        n_tr_data = len(dtrain['R'])
        n_val_data = len(dval['R'])
        n_test_data = len(dtest['R'])
        print("data size: (train,val,test): %i, %i, %i"%(n_tr_data,n_val_data,n_test_data))

        # extract mean and standard deviation of chemical shifts for different atom types
        normalizers = {}
        # normalized_CS = dtrain['CS'].copy()
        for z in shift_types:
            atom_cs = []
            for i in range(n_tr_data):
                atom_cs.extend(dtrain['CS'][i][dtrain['Z'][i] == z])
            normalizers[z] = (np.nanmean(atom_cs), np.nanstd(atom_cs))
        #     for i in range(n_tr_data):
        #         # normalize data for a balanced multi-target training
        #         normalized_CS[i][dtrain['Z'][i] == z] -= normalizers[z][0]
        #         normalized_CS[i][dtrain['Z'][i] == z] /= normalizers[z][1]
        # dtrain['CS_normalized'] = normalized_CS

        atomic_cs_scalers = []
        for i in range(n_tr_data):
            scaler = np.ones_like(dtrain['Z'][i], dtype=float)
            for z in shift_types:
                scaler[dtrain['Z'][i] == z] = normalizers[z][1]
            atomic_cs_scalers.append(scaler)
        dtrain['cs_scaler'] = np.array(atomic_cs_scalers)

    tr_batch_size = settings['training']['tr_batch_size']
    val_batch_size = settings['training']['val_batch_size']
    tr_rotations = settings['training']['tr_rotations']
    val_rotations = settings['training']['val_rotations']

    # generators
    me = settings['general']['driver']

    # steps
    tr_steps = int(np.ceil(n_tr_data / tr_batch_size)) * (tr_rotations + 1)
    val_steps = int(np.ceil(n_val_data / val_batch_size)) * (val_rotations + 1)
    test_steps= int(np.ceil(n_test_data / val_batch_size)) * (val_rotations + 1)

    if settings['data']['pbc']:
        env = PeriodicEnvironment(cutoff=settings['data']['cutoff'] * 1.1)
    else:
        env = ExtensiveEnvironment()

    test_gen = extensive_train_loader(data=dtest,
                                        env_provider=env,
                                        batch_size=val_batch_size,
                                        n_rotations=val_rotations,
                                        freeze_rotations=settings['training']['val_frz_rot'],
                                        keep_original=settings['training']['val_keep_original'],
                                        device=device,
                                        shuffle=False,
                                        drop_last=False)

    if test_only:
        return test_gen, test_steps
    else:
        train_gen = extensive_train_loader(data=dtrain,
                                            env_provider=env,
                                            batch_size=tr_batch_size,
                                            n_rotations=tr_rotations,
                                            freeze_rotations=settings['training']['tr_frz_rot'],
                                            keep_original=settings['training']['tr_keep_original'],
                                            device=device,
                                            shuffle=settings['training']['shuffle'],
                                            drop_last=settings['training']['drop_last'])

        val_gen = extensive_train_loader(data=dval,
                                            env_provider=env,
                                            batch_size=val_batch_size,
                                            n_rotations=val_rotations,
                                            freeze_rotations=settings['training']['val_frz_rot'],
                                            keep_original=settings['training']['val_keep_original'],
                                            device=device,
                                            shuffle=settings['training']['shuffle'],
                                            drop_last=settings['training']['drop_last'])




        return train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, normalizers

def parse_ani_data(settings, device):
    """
    parse and load the ANI-1 dataset with splitting of train, validation and test
    dataset is expected to be in original .h5 format
    (https://figshare.com/collections/_/3846712)
    energy units: Hartree (convert_unit=False), or kcal/mol (convert_unit=True)

    Parameters
    ----------
    settings: instance of yaml file
    device: list
        list of torch devices

    Returns
    -------
    generator: train, val, test generators, respectively
    int: n_steps for train, val, test, respectively
    tuple: tuple of mean and standard deviation of energies in the training data

    """
    from .pyanitools import anidataloader
    atomic_Z_map = {'C': 6, 'H': 1, 'O': 8, 'N': 7}
    # atomic_self_energy = {'H': -0.60467592, 'C': -38.06846167, 'N': -54.70613008, 'O': -75.1796043 } # calculated from dataset
    atomic_self_energy = {'H': -0.500607632585, 'C': -37.8302333826, 'N': -54.5680045287, 'O': -75.0362229210 } # provided by ANI authors
    # meta data
    root = settings['data']['root']
    train_data = settings['data']['train']
    test_data = settings['data']['test']   # can be False

    # Handle train data and test data to make them lists
    if isinstance(train_data, int):
        train_data = [train_data]

    if test_data and isinstance(test_data, int):
        test_data = [test_data]

    test_size_per_molecule = settings['data']['test_size_per_molecule']

    dtrain = {'R':[], 'Z':[], 'N':[], 'E':[]}
    dtest = {'R':[], 'Z':[], 'N':[], 'E':[]}

    for train_data_num in train_data:
        ani_data = anidataloader(root + "/ani_gdb_s0%d.h5" % train_data_num)
        for molecule in ani_data:
            # prepare self energy of the current molecule for subtraction from total energy
            if settings['data']['subtract_self_energy']:
                self_energy = np.sum([atomic_self_energy[a] for a in molecule['species']])
                molecule['energies'] -= self_energy
            if settings['data'].get('convert_unit', True):
                # convert Hartree units to kCal/mol
                molecule['energies'] *= 627.2396
            n_conf, n_atoms, _ = molecule['coordinates'].shape
            conf_indices = np.arange(n_conf)
            # If no test data specified, for each molecule split conformations into train and test
            if test_data is False:
                train_idx, test_idx = train_test_split(conf_indices,
                    test_size=math.ceil(test_size_per_molecule * n_conf),
                    random_state=settings['data']['train_test_split_random_state'])
            else:
                train_idx = conf_indices
            n_conf_train = len(train_idx)
            dtrain['R'].extend(molecule['coordinates'][train_idx])
            dtrain['Z'].extend(np.tile([atomic_Z_map[a] for a in molecule['species']], (n_conf_train, 1)))
            dtrain['N'].extend([n_atoms] * n_conf_train)
            dtrain['E'].extend(molecule['energies'][train_idx])
            if test_data is False:
                n_conf_test = len(test_idx)
                dtest['R'].extend(molecule['coordinates'][test_idx])
                dtest['Z'].extend(np.tile([atomic_Z_map[a] for a in molecule['species']], (n_conf_test, 1)))
                dtest['N'].extend([n_atoms] * n_conf_test)
                dtest['E'].extend(molecule['energies'][test_idx])

    if test_data:
        for test_data_num in test_data:
            ani_data = anidataloader(root + "/ani_gdb_s0%d.h5" % test_data_num)
            for molecule in ani_data:
                 # prepare self energy of the current molecule for subtraction from total energy
                if settings['data']['subtract_self_energy']:
                    self_energy = np.sum([atomic_self_energy[a] for a in molecule['species']])
                    molecule['energies'] -= self_energy
                if settings['data'].get('convert_unit', True):
                    # convert Hartree units to kCal/mol
                    molecule['energies'] *= 627.2396
                n_conf, n_atoms, _ = molecule['coordinates'].shape
                dtest['R'].extend(molecule['coordinates'])
                dtest['Z'].extend(np.tile([atomic_Z_map[a] for a in molecule['species']], (n_conf, 1)))
                dtest['N'].extend([n_atoms] * n_conf)
                dtest['E'].extend(molecule['energies'])

    # Pad irregular-shaped arrays to make all arrays regular in size
    # for k in ['R', 'Z', 'N', 'E']:
    #     dtrain[k] = standardize_batch(dtrain[k])
    #     dtest[k] = standardize_batch(dtest[k])

    further_split_indices = list(range(len(dtrain['R'])))
    train_proportion = settings['data']['train_size_proportion']
    val_proportion = settings['data']['val_size_proportion']
    total_proportion = train_proportion + val_proportion
    if total_proportion < 1:
        train_val_indices, unused_indices = train_test_split(further_split_indices, train_size=total_proportion, random_state=settings['data']['train_val_split_random_state'])
    else:
        train_val_indices = further_split_indices
    train_idx, val_idx = train_test_split(train_val_indices, test_size=(val_proportion / total_proportion), random_state=settings['data']['train_val_split_random_state'])
    data = dtrain
    dtrain = {}
    dval = {}
    for k in data:
        dtrain[k] = np.array(data[k])[train_idx]
        dval[k] = np.array(data[k])[val_idx]
        dtest[k] = np.array(dtest[k])

    # extract data stats
    normalizer = (dtrain['E'].mean(), dtrain['E'].std())

    n_tr_data = len(dtrain['R'])
    n_val_data = len(dval['R'])
    n_test_data = len(dtest['R'])
    print("data size: (train,val,test): %i, %i, %i"%(n_tr_data,n_val_data,n_test_data))

    # HASH check for test energies to make sure test data is fixed
    import hashlib
    test_energy_hash = hashlib.sha1(dtest['E']).hexdigest()
    print("Test set energy HASH:", test_energy_hash)

    tr_batch_size = settings['training']['tr_batch_size']
    val_batch_size = settings['training']['val_batch_size']
    tr_rotations = settings['training']['tr_rotations']
    val_rotations = settings['training']['val_rotations']

    # generators
    me = settings['general']['driver']

    # steps
    tr_steps = int(np.ceil(n_tr_data / tr_batch_size)) * (tr_rotations + 1)
    val_steps = int(np.ceil(n_val_data / val_batch_size)) * (val_rotations + 1)
    test_steps= int(np.ceil(n_test_data / val_batch_size)) * (val_rotations + 1)

    env = ExtensiveEnvironment()

    train_gen = extensive_train_loader(data=dtrain,
                                        env_provider=env,
                                        batch_size=tr_batch_size,
                                        n_rotations=tr_rotations,
                                        freeze_rotations=settings['training']['tr_frz_rot'],
                                        keep_original=settings['training']['tr_keep_original'],
                                        device=device,
                                        shuffle=settings['training']['shuffle'],
                                        drop_last=settings['training']['drop_last'])

    val_gen = extensive_train_loader(data=dval,
                                        env_provider=env,
                                        batch_size=val_batch_size,
                                        n_rotations=val_rotations,
                                        freeze_rotations=settings['training']['val_frz_rot'],
                                        keep_original=settings['training']['val_keep_original'],
                                        device=device,
                                        shuffle=settings['training']['shuffle'],
                                        drop_last=settings['training']['drop_last'])


    test_gen = extensive_train_loader(data=dtest,
                                        env_provider=env,
                                        batch_size=val_batch_size,
                                        n_rotations=val_rotations,
                                        freeze_rotations=settings['training']['val_frz_rot'],
                                        keep_original=settings['training']['val_keep_original'],
                                        device=device,
                                        shuffle=False,
                                        drop_last=False)

    return train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, n_tr_data, n_val_data, n_test_data, normalizer, test_energy_hash

def parse_t1x_data(settings, device):
    """
    implementation based on train and validation size.
    we don't need the test_size in this implementaion.

    Parameters
    ----------
    settings: instance of yaml file
    device: torch.device
        list of torch devices

    Returns
    -------
    generator: train, val, test generators, respectively
    int: n_steps for train, val, test, respectively
    tuple: tuple of mean and standard deviation of energies in the training data

    """
    # meta data
    train_path = settings['data']['train_path']
    val_path = settings['data']['val_path']
    test_path = settings['data']['test_path']   # can be False



    # read data
    dtrain = dict(np.load(train_path))
    dval = dict(np.load(val_path))
    dtest = dict(np.load(test_path))

    # convert unit
    # dtrain['E'] = dtrain['E'] * 23.061
    # dtrain['F'] = dtrain['F'] * 23.061
    # dval['E'] = dval['E'] * 23.061
    # dval['F'] = dval['F'] * 23.061
    # dtest['E'] = dtest['E'] * 23.061
    # dtest['F'] = dtest['F'] * 23.061
    
    # sample data
    if settings['data']['train_size'] != -1:
        n_select = sample_without_replacement(dtrain['R'].shape[0], 
                                              settings['data']['train_size'], 
                                              random_state=settings['data']['random_states'])
        for key in dtrain.keys():
            dtrain[key] = dtrain[key][n_select]
    if settings['data']['val_size'] != -1:
        n_select = sample_without_replacement(dval['R'].shape[0], 
                                              settings['data']['val_size'], 
                                              random_state=settings['data']['random_states'])
        for key in dval.keys():
            dval[key] = dval[key][n_select]
    if settings['data']['test_size'] != -1:
        n_select = sample_without_replacement(dtest['R'].shape[0], 
                                              settings['data']['test_size'], 
                                              random_state=settings['data']['random_states'])
        for key in dtest.keys():
            dtest[key] = dtest[key][n_select]

    # extract data stats
    normalizer = (dtrain['E'].mean(), dtrain['E'].std())

    n_tr_data = dtrain['R'].shape[0]
    n_val_data = dval['R'].shape[0]
    n_test_data = dtest['R'].shape[0]
    print("data size: (train,val,test): %i, %i, %i"%(n_tr_data,n_val_data,n_test_data))

    tr_batch_size = settings['training']['tr_batch_size']
    val_batch_size = settings['training']['val_batch_size']
    tr_rotations = settings['training']['tr_rotations']
    val_rotations = settings['training']['val_rotations']

    # generators
    me = settings['general']['driver']

    # steps
    tr_steps = int(np.ceil(n_tr_data / tr_batch_size)) * (tr_rotations + 1)
    val_steps = int(np.ceil(n_val_data / val_batch_size)) * (val_rotations + 1)
    test_steps = int(np.ceil(n_test_data / val_batch_size)) * (val_rotations + 1)

    env = ExtensiveEnvironment()

    train_gen = extensive_train_loader(data=dtrain,
                                       env_provider=env,
                                       batch_size=tr_batch_size,
                                       n_rotations=tr_rotations,
                                       freeze_rotations=settings['training']['tr_frz_rot'],
                                       keep_original=settings['training']['tr_keep_original'],
                                       device=device,
                                       shuffle=settings['training']['shuffle'],
                                       drop_last=settings['training']['drop_last'])

    val_gen = extensive_train_loader(data=dval,
                                     env_provider=env,
                                     batch_size=val_batch_size,
                                     n_rotations=val_rotations,
                                     freeze_rotations=settings['training']['val_frz_rot'],
                                     keep_original=settings['training']['val_keep_original'],
                                     device=device,
                                     shuffle=settings['training']['shuffle'],
                                     drop_last=settings['training']['drop_last'])

    test_gen = extensive_train_loader(data=dtest,
                                     env_provider=env,
                                     batch_size=val_batch_size,
                                     n_rotations=val_rotations,
                                     freeze_rotations=settings['training']['val_frz_rot'],
                                     keep_original=settings['training']['val_keep_original'],
                                     device=device,
                                     shuffle=False,
                                     drop_last=False)

    return train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, normalizer

def parse_train_test(settings, device, unit='kcal'):
    """
    implementation based on train and validation size.
    we don't need the test_size in this implementaion.

    Parameters
    ----------
    settings: instance of yaml file
    device: torch.device
        list of torch devices

    Returns
    -------
    generator: train, val, test generators, respectively
    int: n_steps for train, val, test, respectively
    tuple: tuple of mean and standard deviation of energies in the training data

    """
    # meta data
    train_path = settings['data']['train_path']
    test_path = settings['data']['test_path']   # can be False

    train_size = settings['data']['train_size']
    val_size = settings['data']['val_size']


    # read data
    data = np.load(train_path)
    test = None
    if test_path:
        test = dict(np.load(test_path))

    # take care of inconsistencies
    dtrain = dict()
    dtest = dict()

    for key in list(data.keys()):
        # copy Z embarrassingly Todo: make it data efficient by flexible environment module
        if key == 'z':
            dtrain['Z'] = np.tile(data['z'], (data['E'].shape[0], 1))
            if test is not None:
                dtest['Z'] = np.tile(test['z'], (test['E'].shape[0], 1))

        elif key == 'E':
            if data['E'].ndim == 1:
                dtrain['E'] = data['E'].reshape(-1,1)
            else:
                dtrain[key] = data[key]

            if test is not None:
                if test['E'].ndim == 1:
                    dtest['E'] = test['E'].reshape(-1, 1)
                else:
                    dtest[key] = test[key]

        elif key in ['R','F','Z']:
            dtrain[key] = data[key]
            if test is not None:
                dtest[key] = test[key]

    # convert unit
    if unit == 'ev':
        dtrain['E'] = dtrain['E'] * 23.061
        dtrain['F'] = dtrain['F'] * 23.061

    # split the data
    dtrain, dval, dtest_leftover = split(dtrain,
                                        train_size=train_size,
                                        test_size=None,
                                        val_size=val_size,
                                        random_states=settings['data']['random_states'])
    if test is None:
        test_size = settings['data'].get('test_size', -1)
        if test_size == -1:
            dtest = dtest_leftover
        else:
            test_size = min(test_size, dtest_leftover['R'].shape[0])
            n_select = sample_without_replacement(dtest_leftover['R'].shape[0],
                                                  test_size,
                                                  random_state=settings['data']['random_states'])
            for k in dtest_leftover.keys():
                dtest[k] = dtest_leftover[k][n_select]


    # extract data stats
    normalizer = (dtrain['E'].mean(), dtrain['E'].std())

    n_tr_data = dtrain['R'].shape[0]
    n_val_data = dval['R'].shape[0]
    n_test_data = dtest['R'].shape[0]
    print("data size: (train,val,test): %i, %i, %i"%(n_tr_data,n_val_data,n_test_data))

    tr_batch_size = settings['training']['tr_batch_size']
    val_batch_size = settings['training']['val_batch_size']
    tr_rotations = settings['training']['tr_rotations']
    val_rotations = settings['training']['val_rotations']

    # generators
    me = settings['general']['driver']

    # steps
    tr_steps = int(np.ceil(n_tr_data / tr_batch_size)) * (tr_rotations + 1)
    val_steps = int(np.ceil(n_val_data / val_batch_size)) * (val_rotations + 1)
    test_steps= int(np.ceil(n_test_data / val_batch_size)) * (val_rotations + 1)

    env = ExtensiveEnvironment()

    train_gen = extensive_train_loader(data=dtrain,
                                       env_provider=env,
                                       batch_size=tr_batch_size,
                                       n_rotations=tr_rotations,
                                       freeze_rotations=settings['training']['tr_frz_rot'],
                                       keep_original=settings['training']['tr_keep_original'],
                                       device=device,
                                       shuffle=settings['training']['shuffle'],
                                       drop_last=settings['training']['drop_last'])

    val_gen = extensive_train_loader(data=dval,
                                     env_provider=env,
                                     batch_size=val_batch_size,
                                     n_rotations=val_rotations,
                                     freeze_rotations=settings['training']['val_frz_rot'],
                                     keep_original=settings['training']['val_keep_original'],
                                     device=device,
                                     shuffle=settings['training']['shuffle'],
                                     drop_last=settings['training']['drop_last'])

    test_gen = extensive_train_loader(data=dtest,
                                     env_provider=env,
                                     batch_size=val_batch_size,
                                     n_rotations=val_rotations,
                                     freeze_rotations=settings['training']['val_frz_rot'],
                                     keep_original=settings['training']['val_keep_original'],
                                     device=device,
                                     shuffle=False,
                                     drop_last=False)

    return train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, normalizer

def parse_methane_data(settings, device):
    """
    parse and load methane combustion reaction by splitting all available data into train, validation and test

    data comes from https://figshare.com/articles/dataset/Dataset_for_methane_combustion/12973055

    reference paper: Zeng, J., Cao, L., Xu, M. et al. Complex reaction processes in combustion unraveled by neural network-based molecular dynamics simulation. Nat Commun 11, 5713 (2020). 
    https://doi.org/10.1038/s41467-020-19497-z

    energy units: eV ?
    force units: eV/A ?

    Parameters
    ----------
    settings: instance of yaml file
    device: torch.device
        list of torch devices

    Returns
    -------
    generator: train, val, test generators, respectively
    int: n_steps for train, val, test, respectively
    tuple: tuple of mean and standard deviation of energies in the training data

    """
    # meta data
    train_path = settings['data']['train_path']
    test_path = settings['data']['test_path']   # can be False
    assert test_path is False


    all_data = os.listdir(train_path)

    data = {'R':[], 'Z':[], 'E':[], 'F':[], 'NA': []}

    type_maps = {0: 6,  #C,
                 1: 1,  #H,
                 2: 8} #O
    atomic_self_energy = [-1034.54661575, -15.63566308, -2043.59323628] # C, H, O

    for composition in all_data:
        types = np.loadtxt(os.path.join(train_path, composition, "type.raw"), dtype=int)
        Z = np.array([type_maps[i] for i in types])
        n_atoms = len(Z)
        coords = np.load(os.path.join(train_path, composition, "set.000", "coord.npy")).reshape((-1, n_atoms, 3))
        energy = np.load(os.path.join(train_path, composition, "set.000", "energy.npy"))
        force = np.load(os.path.join(train_path, composition, "set.000", "force.npy")).reshape((-1, n_atoms, 3))
        # subtract self energy when needed
        if settings['data']['subtract_self_energy']:
            self_energy = np.sum([atomic_self_energy[i] for i in types])
            energy -= self_energy
        n_confs = energy.shape[0]
        Z = np.tile(Z[None], (n_confs, 1))


        # add compositon data into dataset
        data['R'].extend(coords)
        data['Z'].extend(Z)
        data['E'].extend(energy)
        data['F'].extend(force)
        data['NA'].extend([n_atoms] * n_confs)

    # split data into train and test
    all_indices = list(range(len(data['R'])))
    further_split_indices, test_idx = train_test_split(all_indices, test_size=settings['data']['test_count'], random_state=settings['data']['train_test_split_random_state'])
    train_proportion = settings['data']['train_proportion']
    val_proportion = settings['data']['val_proportion']
    total_proportion = train_proportion + val_proportion
    if total_proportion < 1:
        train_val_indices, unused_indices = train_test_split(further_split_indices, train_size=total_proportion, random_state=settings['data']['train_val_split_random_state'])
    else:
        train_val_indices = further_split_indices
    train_idx, val_idx = train_test_split(train_val_indices, test_size=(val_proportion / total_proportion), random_state=settings['data']['train_val_split_random_state'])

    dtrain = {}
    dval = {}
    dtest = {}
    for k in data:
        dtrain[k] = np.array(data[k])[train_idx]
        dval[k] = np.array(data[k])[val_idx]
        dtest[k] = np.array(data[k])[test_idx]


    # extract data stats
    normalizer = (dtrain['E'].mean(), dtrain['E'].std())

    n_tr_data = dtrain['R'].shape[0]
    n_val_data = dval['R'].shape[0]
    n_test_data = dtest['R'].shape[0]
    print("data size: (train,val,test): %i, %i, %i"%(n_tr_data,n_val_data,n_test_data))

    # HASH check for test energies to make sure test data is fixed
    import hashlib
    test_energy_hash = hashlib.sha1(dtest['E']).hexdigest()
    print("Test set energy HASH:", test_energy_hash)

    tr_batch_size = settings['training']['tr_batch_size']
    val_batch_size = settings['training']['val_batch_size']
    tr_rotations = settings['training']['tr_rotations']
    val_rotations = settings['training']['val_rotations']

    # generators
    me = settings['general']['driver']

    # steps
    tr_steps = int(np.ceil(n_tr_data / tr_batch_size)) * (tr_rotations + 1)
    val_steps = int(np.ceil(n_val_data / val_batch_size)) * (val_rotations + 1)
    test_steps= int(np.ceil(n_test_data / val_batch_size)) * (val_rotations + 1)

    env = ExtensiveEnvironment()

    train_gen = extensive_train_loader(data=dtrain,
                                       env_provider=env,
                                       batch_size=tr_batch_size,
                                       n_rotations=tr_rotations,
                                       freeze_rotations=settings['training']['tr_frz_rot'],
                                       keep_original=settings['training']['tr_keep_original'],
                                       device=device,
                                       shuffle=settings['training']['shuffle'],
                                       drop_last=settings['training']['drop_last'])

    val_gen = extensive_train_loader(data=dval,
                                     env_provider=env,
                                     batch_size=val_batch_size,
                                     n_rotations=val_rotations,
                                     freeze_rotations=settings['training']['val_frz_rot'],
                                     keep_original=settings['training']['val_keep_original'],
                                     device=device,
                                     shuffle=settings['training']['shuffle'],
                                     drop_last=settings['training']['drop_last'])

    test_gen = extensive_train_loader(data=dtest,
                                     env_provider=env,
                                     batch_size=val_batch_size,
                                     n_rotations=val_rotations,
                                     freeze_rotations=settings['training']['val_frz_rot'],
                                     keep_original=settings['training']['val_keep_original'],
                                     device=device,
                                     shuffle=False,
                                     drop_last=False)

    return train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, n_tr_data, n_val_data, n_test_data, normalizer, test_energy_hash
