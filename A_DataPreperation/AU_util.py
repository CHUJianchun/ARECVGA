import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import trange, tqdm
import glob
import gzip
from args import args
from A_DataPreperation.AU_dataset import ProcessedDataset

QM_structure_path = 'DataA_InputData/gdb9.sdf'
QM_property_path = 'DataA_InputData/gdb9.sdf.csv'
ZINC_structure_path = 'DataA_InputData/ZINC'
Electrolyte_SMILES_path = 'DataA_InputData/electrolyte.txt'
atom_path = 'DataA_InputData/atomref.txt'
zinc_charge_dict = {'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16}
qm9_charge_dict = {'C': 6, 'N': 7, 'O': 8, 'F': 9}
ZINC_CHARGE_LIST = [6, 7, 8, 9, 15, 16]
QM9_CHARGE_LIST = [6, 7, 8, 9]
max_length = 1.
mol_list = []

score_matrix = np.array(
    [[5, 2, 5, 3.5, 4, 5, 1],
     [3.5, 4, 4, 2, 4.5, 3, 5],
     [3, 3, 4, 3, 3.75, 5, 5],
     [3, 2, 3, 4, 5, 2, 3],
     [5, 1.5, 3, 5, 4, 3, 4],
     [4, 1, 1, 3, 5, 2, 2],
     [4, 2, 1, 3, 4, 1, 5]]
) / 5


def score(mol):
    cyclic_ester_flag = mol.HasSubstructMatch(Chem.MolFromSmiles('C1COC(=O)O1')) and 'F' in Chem.MolToSmiles(mol)
    linear_ester_flag = mol.HasSubstructMatch(Chem.MolFromSmiles('COC(=O)OC')) and 'F' in Chem.MolToSmiles(mol)
    cyclic_ester_ids = (100, 101)
    linear_ester_ids = (100, 101)
    if cyclic_ester_flag:
        cyclic_ester_ids = mol.GetSubstructMatch(Chem.MolFromSmiles('C1COC(=O)O1'))
    elif linear_ester_flag:
        linear_ester_ids = mol.GetSubstructMatch(Chem.MolFromSmiles('COC(=O)OC'))

    ether_flag = (mol.HasSubstructMatch(Chem.MolFromSmiles('OCCO')) or mol.HasSubstructMatch(Chem.MolFromSmiles('O1CCCC1'))) and 'F' in Chem.MolToSmiles(mol)
    if cyclic_ester_flag or linear_ester_flag:
        coc_ids_array = np.array(mol.GetSubstructMatches(Chem.MolFromSmiles('COC')))
        if len(coc_ids_array.shape) != 1:
            coc_o_id = coc_ids_array[:, 1]
            if (cyclic_ester_flag and [x for x in coc_o_id if x not in cyclic_ester_ids]) or (
                    linear_ester_flag and [x for x in coc_o_id if x not in linear_ester_ids]):
                ether_flag = True

    sulfone_flag = mol.HasSubstructMatch(Chem.MolFromSmiles('CS(=O)(=O)C'))
    phosphate_flag = mol.HasSubstructMatch(Chem.MolFromSmiles('OP(=O)(O)O')) or mol.HasSubstructMatch(Chem.MolFromSmiles('OP(O)O'))
    nitrile_flag = mol.HasSubstructMatch(Chem.MolFromSmiles('C#N')) and 'F' in Chem.MolToSmiles(mol)
    carbamate_flag = mol.HasSubstructMatch(Chem.MolFromSmiles('NC(=O)OC')) and 'F' in Chem.MolToSmiles(mol)
    flag_array = np.array([cyclic_ester_flag, linear_ester_flag, ether_flag, sulfone_flag, phosphate_flag, nitrile_flag, carbamate_flag])
    has_label = any(flag_array)
    if not has_label:
        return None
    else:
        return np.matmul(flag_array, score_matrix) / flag_array.sum()


def get_mol_list():
    mol_list____ = []
    for filename in tqdm(glob.glob(r'DataA_InputData\ZINC\**\**\*.sdf')):
        structure_file = Chem.SDMolSupplier(filename, removeHs=True)
        for i in range(len(structure_file)):
            mol_list____.append(structure_file[i])

    return mol_list____


def unzip():
    for filename in glob.glob(r'DataA_InputData\ZINC\**\**\*.gz'):
        f_name = filename.replace('.gz', '')
        g_file = gzip.GzipFile(filename)
        open(f_name, 'wb+').write(g_file.read())
        g_file.close()


def atom_in_dict(charge_list):
    for i in charge_list:
        if i not in ZINC_CHARGE_LIST:
            return False
    return True


def prepare_ZINC_dataset_npz(atom_range=(7, args.zinc_padding_dim)):
    mol_list_ = get_mol_list()

    padding_dim = args.zinc_padding_dim
    position_list = []
    adjacency_list = []
    adjacency_feature_list = []
    num_atom_list = []
    charge_list = []
    charge_onehot_list = []
    smiles_list = []
    explicit_h_list = []
    score_list = []
    molecule_num = 0   ###
    for i in trange(len(mol_list_)):
        try:
            mol = mol_list_[i]
            a = mol.GetNumAtoms()
        except:
            continue
        else:
            mol_coordinate = []
            Chem.Kekulize(mol)
            smiles = Chem.MolToSmiles(mol)
            if atom_range[0] <= mol.GetNumAtoms() <= atom_range[1]:
                mol_charge = []
                explicit_h = np.zeros((args.zinc_padding_dim, 4)).tolist()
                for j in range(mol.GetNumAtoms()):
                    mol_coordinate.append(list(mol.GetConformer().GetAtomPosition(j)))
                    mol_charge.append(mol.GetAtomWithIdx(j).GetAtomicNum())
                    eh = [0, 0, 0, 0]
                    eh[mol.GetAtomWithIdx(j).GetNumExplicitHs()] = 1
                    explicit_h[j][mol.GetAtomWithIdx(j).GetNumExplicitHs() + mol.GetAtomWithIdx(j).GetNumImplicitHs()] = 1
                mol_score = score(mol)
                if atom_in_dict(mol_charge) and mol_score is not None:
                    charge_list.append(mol_charge)
                    explicit_h_list.append(explicit_h)
                    adjacency_list.append(Chem.rdmolops.GetAdjacencyMatrix(mol))
                    adjacency_feature = np.zeros((padding_dim, padding_dim, 5))
                    adjacency_feature[:, :, 3] = 1
                    adjacency_feature[mol.GetNumAtoms():, :, 4] = max_length
                    adjacency_feature[:, mol.GetNumAtoms():, 4] = max_length
                    adjacency_feature[mol.GetNumAtoms():, mol.GetNumAtoms():, 4] = 0
                    for j in range(mol.GetNumBonds()):
                        bond = mol.GetBondWithIdx(j)
                        adjacency_feature[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type(bond)] = 1
                        adjacency_feature[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), bond_type(bond)] = 1
                        adjacency_feature[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), 3] = 0
                        adjacency_feature[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), 3] = 0
                    for j in range(mol.GetNumAtoms()):
                        for k in range(j, mol.GetNumAtoms()):
                            x1, y1, z1 = mol.GetConformer().GetAtomPosition(j)
                            x2, y2, z2 = mol.GetConformer().GetAtomPosition(k)
                            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5 / 2
                            adjacency_feature[j, k, 4] = distance
                            adjacency_feature[k, j, 4] = distance
                    num_atom_list.append(mol.GetNumAtoms())
                    adjacency_feature_list.append(adjacency_feature)
                    position_list.append(mol_coordinate)
                    smiles_list.append(Chem.MolToSmiles(mol))
                    score_list.append(mol_score)
                    molecule_num += 1
                    if 'P' in Chem.MolToSmiles(mol):
                        for times in range(400):
                            charge_list.append(mol_charge)
                            explicit_h_list.append(explicit_h)
                            adjacency_list.append(Chem.rdmolops.GetAdjacencyMatrix(mol))
                            adjacency_feature = np.zeros((padding_dim, padding_dim, 5))
                            adjacency_feature[:, :, 3] = 1
                            adjacency_feature[mol.GetNumAtoms():, :, 4] = max_length
                            adjacency_feature[:, mol.GetNumAtoms():, 4] = max_length
                            adjacency_feature[mol.GetNumAtoms():, mol.GetNumAtoms():, 4] = 0
                            for j in range(mol.GetNumBonds()):
                                bond = mol.GetBondWithIdx(j)
                                adjacency_feature[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type(bond)] = 1
                                adjacency_feature[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), bond_type(bond)] = 1
                                adjacency_feature[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), 3] = 0
                                adjacency_feature[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), 3] = 0
                            for j in range(mol.GetNumAtoms()):
                                for k in range(j, mol.GetNumAtoms()):
                                    x1, y1, z1 = mol.GetConformer().GetAtomPosition(j)
                                    x2, y2, z2 = mol.GetConformer().GetAtomPosition(k)
                                    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5 / 2
                                    adjacency_feature[j, k, 4] = distance
                                    adjacency_feature[k, j, 4] = distance
                            num_atom_list.append(mol.GetNumAtoms())
                            adjacency_feature_list.append(adjacency_feature)
                            position_list.append(mol_coordinate)
                            smiles_list.append(Chem.MolToSmiles(mol))
                            score_list.append(mol_score)
                    if 'S' in Chem.MolToSmiles(mol):
                        for times in range(30):
                            charge_list.append(mol_charge)
                            explicit_h_list.append(explicit_h)
                            adjacency_list.append(Chem.rdmolops.GetAdjacencyMatrix(mol))
                            adjacency_feature = np.zeros((padding_dim, padding_dim, 5))
                            adjacency_feature[:, :, 3] = 1
                            adjacency_feature[mol.GetNumAtoms():, :, 4] = max_length
                            adjacency_feature[:, mol.GetNumAtoms():, 4] = max_length
                            adjacency_feature[mol.GetNumAtoms():, mol.GetNumAtoms():, 4] = 0
                            for j in range(mol.GetNumBonds()):
                                bond = mol.GetBondWithIdx(j)
                                adjacency_feature[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type(bond)] = 1
                                adjacency_feature[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), bond_type(bond)] = 1
                                adjacency_feature[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), 3] = 0
                                adjacency_feature[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), 3] = 0
                            for j in range(mol.GetNumAtoms()):
                                for k in range(j, mol.GetNumAtoms()):
                                    x1, y1, z1 = mol.GetConformer().GetAtomPosition(j)
                                    x2, y2, z2 = mol.GetConformer().GetAtomPosition(k)
                                    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5 / 2
                                    adjacency_feature[j, k, 4] = distance
                                    adjacency_feature[k, j, 4] = distance
                            num_atom_list.append(mol.GetNumAtoms())
                            adjacency_feature_list.append(adjacency_feature)
                            position_list.append(mol_coordinate)
                            smiles_list.append(Chem.MolToSmiles(mol))
                            score_list.append(mol_score)

    edge_feature_list = np.zeros((len(charge_list), padding_dim ** 2, 5))
    edge_feature_list[:, :, 3] = 1
    for i in trange(len(charge_list)):
        charge_list[i] = np.pad(np.array(charge_list[i]), (0, padding_dim - len(charge_list[i])), 'constant')
        charge_onehot = np.zeros((padding_dim, 7))
        for j in range(padding_dim):
            if charge_list[i][j] == 6:
                charge_onehot[j, 0] = 1.
            elif charge_list[i][j] == 7:
                charge_onehot[j, 1] = 1.
            elif charge_list[i][j] == 8:
                charge_onehot[j, 2] = 1.
            elif charge_list[i][j] == 9:
                charge_onehot[j, 3] = 1.
            elif charge_list[i][j] == 15:
                charge_onehot[j, 4] = 1.
            elif charge_list[i][j] == 16:
                charge_onehot[j, 5] = 1.
            else:
                charge_onehot[j, 6] = 1.
        charge_onehot_list.append(charge_onehot)
        adjacency_list[i] = np.pad(np.array(adjacency_list[i]),
                                   ((0, padding_dim - len(adjacency_list[i][0])),
                                    (0, padding_dim - len(adjacency_list[i][0]))),
                                   'constant')
        position_list[i] = np.pad(np.array(position_list[i]),
                                  ((0, padding_dim - len(position_list[i])), (0, 0)),
                                  'constant')
        for j in range(padding_dim):
            for k in range(padding_dim):
                edge_feature_list[i][j * padding_dim + k] = adjacency_feature_list[i][j, k]
    charge_list = np.array(charge_list, dtype='int')
    explicit_h_list = np.array(explicit_h_list, dtype='int')
    charge_onehot_list = np.array(charge_onehot_list)
    adjacency_list = np.array(adjacency_list, dtype='float32')
    num_atom_list = np.array(num_atom_list, dtype='int').reshape(-1, 1)
    position_list = np.array(position_list, dtype='float32')
    smiles_list = np.array(smiles_list)
    score_list = np.array(score_list)
    for i__ in range(adjacency_list.shape[0]):
        atom_num = int(num_atom_list[i__][0])
        adjacency_list[i__] = adjacency_list[i__] + np.eye(padding_dim)
        adjacency_list[i__, atom_num:, atom_num:] = 1.
    out_file_path = 'DataB_ProcessedData/ZINC_All_Mol.npz'
    np.savez(out_file_path,
             smiles_list=smiles_list,
             charge_list=charge_list,
             adjacency_list=adjacency_list,
             num_atom_list=num_atom_list,
             position_list=position_list,
             charge_onehot_list=charge_onehot_list,
             edge_feature_list=edge_feature_list,
             explicit_h_list=explicit_h_list,
             score_list=score_list)


def prepare_optimization_init_dataset_npz(atom_range=(1, 35)):
    padding_dim = args.padding_dim
    with open(Electrolyte_SMILES_path, 'r') as txt_component_smiles:
        component_smiles = txt_component_smiles.readlines()
    structure_file = [Chem.MolFromSmiles(cs[:-1]) for cs in component_smiles]
    position_list = []
    adjacency_list = []
    adjacency_feature_list = []
    num_atom_list = []
    charge_list = []
    charge_onehot_list = []

    for i in trange(len(structure_file)):
        mol_coordinate = []
        try:
            mol = structure_file[i]
            num_atom_list.append(mol.GetNumAtoms())
        except:
            pass
        else:
            Chem.Kekulize(mol)
            AllChem.EmbedMolecule(mol)
            if atom_range[0] <= mol.GetNumAtoms() <= atom_range[1]:
                mol_charge = []
                for j in range(mol.GetNumAtoms()):
                    mol_coordinate.append(list(mol.GetConformer().GetAtomPosition(j)))
                    mol_charge.append(mol.GetAtomWithIdx(j).GetAtomicNum())
                if atom_in_dict(mol_charge):
                    charge_list.append(mol_charge)
                else:
                    continue
                adjacency_list.append(Chem.rdmolops.GetAdjacencyMatrix(mol))
                adjacency_feature = np.zeros((padding_dim, padding_dim, 5))
                adjacency_feature[:, :, 3] = 1
                adjacency_feature[mol.GetNumAtoms():, :, 4] = max_length
                adjacency_feature[:, mol.GetNumAtoms():, 4] = max_length
                adjacency_feature[mol.GetNumAtoms():, mol.GetNumAtoms():, 4] = 0
                for j in range(mol.GetNumBonds()):
                    bond = mol.GetBondWithIdx(j)
                    adjacency_feature[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type(bond)] = 1
                    adjacency_feature[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), bond_type(bond)] = 1
                    adjacency_feature[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), 3] = 0
                    adjacency_feature[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), 3] = 0
                for j in range(mol.GetNumAtoms()):
                    for k in range(j, mol.GetNumAtoms()):
                        x1, y1, z1 = mol.GetConformer().GetAtomPosition(j)
                        x2, y2, z2 = mol.GetConformer().GetAtomPosition(k)
                        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5 / 2
                        adjacency_feature[j, k, 4] = distance
                        adjacency_feature[k, j, 4] = distance
                adjacency_feature_list.append(adjacency_feature)
                position_list.append(mol_coordinate)
    del mol_coordinate, mol_charge, structure_file
    edge_feature_list = np.zeros((len(charge_list), padding_dim ** 2, 5))
    edge_feature_list[:, :, 3] = 1
    for i in trange(len(charge_list)):
        charge_list[i] = np.pad(np.array(charge_list[i]), (0, padding_dim - len(charge_list[i])), 'constant')
        charge_onehot = np.zeros((padding_dim, 10))
        for j in range(padding_dim):
            if charge_list[i][j] == 1:
                charge_onehot[j, 0] = 1.
            elif charge_list[i][j] == 5:
                charge_onehot[j, 1] = 1.
            elif charge_list[i][j] == 6:
                charge_onehot[j, 2] = 1.
            elif charge_list[i][j] == 7:
                charge_onehot[j, 3] = 1.
            elif charge_list[i][j] == 8:
                charge_onehot[j, 4] = 1.
            elif charge_list[i][j] == 9:
                charge_onehot[j, 5] = 1.
            elif charge_list[i][j] == 15:
                charge_onehot[j, 6] = 1.
            elif charge_list[i][j] == 16:
                charge_onehot[j, 7] = 1.
            elif charge_list[i][j] == 17:
                charge_onehot[j, 8] = 1.
            else:
                charge_onehot[j, 9] = 1.
        charge_onehot_list.append(charge_onehot)
        adjacency_list[i] = np.pad(np.array(adjacency_list[i]),
                                   ((0, padding_dim - len(adjacency_list[i][0])),
                                    (0, padding_dim - len(adjacency_list[i][0]))),
                                   'constant')
        position_list[i] = np.pad(np.array(position_list[i]),
                                  ((0, padding_dim - len(position_list[i])), (0, 0)),
                                  'constant')
        for j in range(padding_dim):
            for k in range(padding_dim):
                edge_feature_list[i][j * padding_dim + k] = adjacency_feature_list[i][j, k]
    charge_list = np.array(charge_list, dtype='int')
    charge_onehot_list = np.array(charge_onehot_list)
    adjacency_list = np.array(adjacency_list, dtype='float32')
    num_atom_list = np.array(num_atom_list, dtype='int').reshape(-1, 1)
    position_list = np.array(position_list, dtype='float32')
    for i__ in range(adjacency_list.shape[0]):
        atom_num = int(num_atom_list[i__][0])
        adjacency_list[i__] = adjacency_list[i__] + np.eye(args.padding_dim)
        adjacency_list[i__, atom_num:, atom_num:] = 1.
    out_file_path = 'DataB_ProcessedData/Electrolyte_All_Mol.npz'
    np.savez(out_file_path,
             charge_list=charge_list,
             adjacency_list=adjacency_list,
             num_atom_list=num_atom_list,
             position_list=position_list,
             charge_onehot_list=charge_onehot_list,
             edge_feature_list=edge_feature_list)


def SmilesToData(component_smiles, atom_range=(5, args.zinc_padding_dim), out_file_path=None):
    padding_dim = args.padding_dim
    structure_file = [Chem.MolFromSmiles(cs) for cs in component_smiles]
    position_list = []
    adjacency_list = []
    adjacency_feature_list = []
    num_atom_list = []
    charge_list = []
    charge_onehot_list = []
    explicit_h_list = []
    for i in trange(len(structure_file)):
        mol_coordinate = []
        try:
            mol = structure_file[i]
            num_atom_list.append(mol.GetNumAtoms())
        except:
            pass
        else:
            Chem.Kekulize(mol)
            AllChem.EmbedMolecule(mol)
            if atom_range[0] <= mol.GetNumAtoms() <= atom_range[1]:
                mol_charge = []
                explicit_h = np.zeros((args.zinc_padding_dim, 4)).tolist()
                for j in range(mol.GetNumAtoms()):
                    mol_coordinate.append(list(mol.GetConformer().GetAtomPosition(j)))
                    mol_charge.append(mol.GetAtomWithIdx(j).GetAtomicNum())
                    eh = [0, 0, 0, 0]
                    eh[mol.GetAtomWithIdx(j).GetNumExplicitHs()] = 1
                    explicit_h[j][mol.GetAtomWithIdx(j).GetNumExplicitHs() + mol.GetAtomWithIdx(j).GetNumImplicitHs()] = 1
                if atom_in_dict(mol_charge):
                    charge_list.append(mol_charge)
                    explicit_h_list.append(explicit_h)
                else:
                    continue
                adjacency_list.append(Chem.rdmolops.GetAdjacencyMatrix(mol))
                adjacency_feature = np.zeros((padding_dim, padding_dim, 5))
                adjacency_feature[:, :, 3] = 1
                adjacency_feature[mol.GetNumAtoms():, :, 4] = max_length
                adjacency_feature[:, mol.GetNumAtoms():, 4] = max_length
                adjacency_feature[mol.GetNumAtoms():, mol.GetNumAtoms():, 4] = 0
                for j in range(mol.GetNumBonds()):
                    bond = mol.GetBondWithIdx(j)
                    adjacency_feature[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type(bond)] = 1
                    adjacency_feature[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), bond_type(bond)] = 1
                    adjacency_feature[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), 3] = 0
                    adjacency_feature[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), 3] = 0
                for j in range(mol.GetNumAtoms()):
                    for k in range(j, mol.GetNumAtoms()):
                        x1, y1, z1 = mol.GetConformer().GetAtomPosition(j)
                        x2, y2, z2 = mol.GetConformer().GetAtomPosition(k)
                        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5 / 2
                        adjacency_feature[j, k, 4] = distance
                        adjacency_feature[k, j, 4] = distance
                adjacency_feature_list.append(adjacency_feature)
                position_list.append(mol_coordinate)
    del mol_coordinate, mol_charge, structure_file
    edge_feature_list = np.zeros((len(charge_list), padding_dim ** 2, 5))
    edge_feature_list[:, :, 3] = 1
    for i in trange(len(charge_list)):
        charge_list[i] = np.pad(np.array(charge_list[i]), (0, padding_dim - len(charge_list[i])), 'constant')
        charge_onehot = np.zeros((padding_dim, 9))
        for j in range(padding_dim):
            if charge_list[i][j] == 5:
                charge_onehot[j, 0] = 1.
            elif charge_list[i][j] == 6:
                charge_onehot[j, 1] = 1.
            elif charge_list[i][j] == 7:
                charge_onehot[j, 2] = 1.
            elif charge_list[i][j] == 8:
                charge_onehot[j, 3] = 1.
            elif charge_list[i][j] == 9:
                charge_onehot[j, 4] = 1.
            elif charge_list[i][j] == 15:
                charge_onehot[j, 5] = 1.
            elif charge_list[i][j] == 16:
                charge_onehot[j, 6] = 1.
            elif charge_list[i][j] == 17:
                charge_onehot[j, 7] = 1.
            else:
                charge_onehot[j, 8] = 1.
        charge_onehot_list.append(charge_onehot)
        adjacency_list[i] = np.pad(np.array(adjacency_list[i]),
                                   ((0, padding_dim - len(adjacency_list[i][0])),
                                    (0, padding_dim - len(adjacency_list[i][0]))),
                                   'constant')
        position_list[i] = np.pad(np.array(position_list[i]),
                                  ((0, padding_dim - len(position_list[i])), (0, 0)),
                                  'constant')
        for j in range(padding_dim):
            for k in range(padding_dim):
                edge_feature_list[i][j * padding_dim + k] = adjacency_feature_list[i][j, k]
    charge_list = np.array(charge_list, dtype='int')
    charge_onehot_list = np.array(charge_onehot_list)
    adjacency_list = np.array(adjacency_list, dtype='float32')
    num_atom_list = np.array(num_atom_list, dtype='int').reshape(-1, 1)
    position_list = np.array(position_list, dtype='float32')
    for i__ in range(adjacency_list.shape[0]):
        atom_num = int(num_atom_list[i__][0])
        adjacency_list[i__] = adjacency_list[i__] + np.eye(args.padding_dim)
        adjacency_list[i__, atom_num:, atom_num:] = 1.
    if out_file_path is not None:
        np.savez(out_file_path,
                 charge_list=charge_list,
                 adjacency_list=adjacency_list,
                 num_atom_list=num_atom_list,
                 position_list=position_list,
                 charge_onehot_list=charge_onehot_list,
                 edge_feature_list=edge_feature_list)
    return {'charge_list': charge_list,
            'adjacency_list': adjacency_list,
            'num_atom_list': num_atom_list,
            'position_list': position_list,
            'charge_onehot_list': charge_onehot_list,
            'edge_feature_list': edge_feature_list}


def prepare_Augment_dataset_npz():
    def save_list(atom, output_path):
        for i_ in range(8):
            dataset_list = []
            dataset_ = ZINC_dataset_(i_, tensor=False)
            for data_ in tqdm(dataset_):
                if atom in data_['smiles_list']:
                    dataset_list.append(data_)
            smiles_list = []
            charge_list = []
            adjacency_list = []
            num_atom_list = []
            position_list = []
            charge_onehot_list = []
            edge_feature_list = []
            for data_ in dataset_list:
                smiles_list.append(data_['smiles_list'])
                charge_list.append(data_['charge_list'])
                adjacency_list.append(data_['adjacency_list'])
                num_atom_list.append(data_['num_atom_list'])
                position_list.append(data_['position_list'])
                charge_onehot_list.append(data_['charge_onehot_list'])
                edge_feature_list.append(data_['edge_feature_list'])
            charge_list = np.array(charge_list, dtype='int')
            charge_onehot_list = np.array(charge_onehot_list)
            adjacency_list = np.array(adjacency_list, dtype='float32')
            num_atom_list = np.array(num_atom_list, dtype='int').reshape(-1, 1)
            position_list = np.array(position_list, dtype='float32')
            smiles_list = np.array(smiles_list)
            edge_feature_list = np.array(edge_feature_list)
            np.savez(output_path[0] + str(i_) + output_path[1],
                     smiles_list=smiles_list,
                     charge_list=charge_list,
                     adjacency_list=adjacency_list,
                     num_atom_list=num_atom_list,
                     position_list=position_list,
                     charge_onehot_list=charge_onehot_list,
                     edge_feature_list=edge_feature_list)

    def cat(output_path):
        cat_path = output_path[0] + 'CAT' + output_path[1]
        smiles_list = []
        charge_list = []
        adjacency_list = []
        num_atom_list = []
        position_list = []
        charge_onehot_list = []
        edge_feature_list = []
        for i_ in trange(8):
            dataset = np.load(output_path[0] + str(i_) + output_path[1])
            if dataset['smiles_list'].shape[0] != 0:
                smiles_list.append(dataset['smiles_list'])
                charge_list.append(dataset['charge_list'])
                adjacency_list.append(dataset['adjacency_list'])
                num_atom_list.append(dataset['num_atom_list'])
                position_list.append(dataset['position_list'])
                charge_onehot_list.append(dataset['charge_onehot_list'])
                edge_feature_list.append(dataset['edge_feature_list'])
        smiles_list = np.hstack(smiles_list)
        charge_list = np.vstack(charge_list)
        adjacency_list = np.vstack(adjacency_list)
        num_atom_list = np.vstack(num_atom_list)
        position_list = np.vstack(position_list)
        charge_onehot_list = np.vstack(charge_onehot_list)
        edge_feature_list = np.vstack(edge_feature_list)
        np.savez(cat_path,
                 smiles_list=smiles_list,
                 charge_list=charge_list,
                 adjacency_list=adjacency_list,
                 num_atom_list=num_atom_list,
                 position_list=position_list,
                 charge_onehot_list=charge_onehot_list,
                 edge_feature_list=edge_feature_list)

    f_output_path = ['DataB_ProcessedData/ZINC_All_Mol_Aug_F_', '.npz']
    p_output_path = ['DataB_ProcessedData/ZINC_All_Mol_Aug_P_', '.npz']
    s_output_path = ['DataB_ProcessedData/ZINC_All_Mol_Aug_S_', '.npz']
    cl_output_path = ['DataB_ProcessedData/ZINC_All_Mol_Aug_Cl_', '.npz']

    save_list('F', f_output_path)
    save_list('P', p_output_path)
    save_list('S', s_output_path)
    save_list('Cl', cl_output_path)
    cat(f_output_path)
    cat(p_output_path)
    cat(s_output_path)
    cat(cl_output_path)


def prepare_pre_train_dataset_npz():
    dataset_ = ZINC_dataset_(tensor=False)
    padding_dim = args.zinc_padding_dim
    index = []
    for i in trange(len(dataset_)):
        if 'F' in dataset_.data['smiles_list'][i]:
            index.append(i)
        if 'P' in dataset_.data['smiles_list'][i]:
            index.append(i)
            index.append(i)
    pre_train_dataset_ = dataset_[index]
    out_file_path = 'DataB_ProcessedData/ZINC_Pre_Train.npz'
    np.savez(out_file_path,
             smiles_list=pre_train_dataset_['smiles_list'],
             charge_list=pre_train_dataset_['charge_list'],
             adjacency_list=pre_train_dataset_['adjacency_list'],
             num_atom_list=pre_train_dataset_['num_atom_list'],
             position_list=pre_train_dataset_['position_list'],
             charge_onehot_list=pre_train_dataset_['charge_onehot_list'],
             edge_feature_list=pre_train_dataset_['edge_feature_list'],
             explicit_h_list=pre_train_dataset_['explicit_h_list'],
             score_list=pre_train_dataset_['score_list'])


def QM_dataset_(path_='DataB_ProcessedData/QM_All_Mol.npz', return_path=False):
    data = np.load(path_)
    if return_path:
        return ProcessedDataset(data), path_
    else:
        return ProcessedDataset(data)


def Pre_train_dataset_(return_path=False, tensor=True):
    path_ = 'DataB_ProcessedData/ZINC_Pre_Train.npz'
    data = np.load(path_)
    if return_path:
        return ProcessedDataset(data, tensor), path_
    else:
        return ProcessedDataset(data, tensor)


def ZINC_dataset_(return_path=False, tensor=True):
    path_ = 'DataB_ProcessedData/ZINC_All_Mol.npz'
    data = np.load(path_)
    if return_path:
        return ProcessedDataset(data, tensor), path_
    else:
        return ProcessedDataset(data, tensor)


def ELE_dataset_(path_='DataB_ProcessedData/Electrolyte_All_Mol.npz', repeat_time=1):
    data = np.load(path_)
    return ProcessedDataset(data, repeat=repeat_time)


def Aug_dataset_(atom=None, i__=None, repeat_time=1):
    atom_list = ['F', 'P', 'S', 'Cl']
    if i__ is None:
        data = np.load('DataB_ProcessedData/ZINC_All_Mol_Aug_' + atom + '_CAT.npz')
    elif atom is None:
        data = np.load('DataB_ProcessedData/ZINC_All_Mol_Aug_' + atom_list[i__] + '_CAT.npz')
    else:
        raise ValueError('Input of Aug_dataset_ not valid')
    return ProcessedDataset(data, repeat=repeat_time)


def bond_type(bond):
    if bond.GetBondType().name == "SINGLE":
        return 0
    elif bond.GetBondType().name == "DOUBLE":
        return 1
    elif bond.GetBondType().name == "TRIPLE":
        return 2
    else:
        raise ValueError(bond.GetBondType())
