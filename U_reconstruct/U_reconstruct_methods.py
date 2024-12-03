import torch
import numpy as np
import time
import rdkit
import rdkit.Chem as Chem
from args import args
from B_Train.BU_util import *
import torch.nn.functional as f

ZINC_charge_dict = {'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16}
ZINC_h_index = {'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 'S': 6}
ZINC_CHARGE_LIST = [6, 7, 8, 9, 15, 16]


def num_to_bond_type(num):
    if num == 1:
        bond = Chem.BondType.SINGLE
    elif num == 2:
        bond = Chem.BondType.DOUBLE
    elif num == 3:
        bond = Chem.BondType.TRIPLE
    else:
        raise ValueError(num)
    return bond


def ZINC_index_to_rdkit_atom(index):
    atom_index = {'0': 6, '1': 7, '2': 8, '3': 9, '4': 15, '5': 16}
    return Chem.Atom(atom_index[str(index)])


def ZINC_index_to_atom(index, hs):
    atom_index = {'0': (6, 4 - hs if 4 - hs > 0 else 1),  # C
                  '1': (7, 3 - hs if 3 - hs > 0 else 1),  # N 3
                  '2': (8, 2 - hs if 2 - hs > 0 else 1),  # O
                  '3': (9, 1 - hs if 1 - hs > 0 else 1),  # F
                  '4': (15, 6 - hs if 6 - hs > 0 else 1),  # P 6
                  '5': (16, 6 - hs if 6 - hs > 0 else 1),  # S 2 或 4 或 6
                  }
    return Atom(atom_index[str(index.cpu().numpy() if isinstance(index, torch.Tensor) else index)])


def rdkit_mol(mol):
    try:
        r_mol = Chem.RWMol(Chem.MolFromSmiles(''))
        for i in range(args.zinc_padding_dim):
            if mol.atom_list[i] is not None:
                r_mol.AddAtom(mol.atom_list[i].atom)
        for bond_start in range(args.zinc_padding_dim):
            for bond_end in range(bond_start, args.zinc_padding_dim):
                if mol.adjacency_matrix[bond_start, bond_end] != 0:
                    r_mol.AddBond(bond_start, bond_end, num_to_bond_type(mol.adjacency_matrix[bond_start, bond_end]))
        return r_mol
    except RuntimeError:
        return Chem.RWMol(Chem.MolFromSmiles(''))


class Molecule:
    def __init__(self):
        self.atom_list = [None] * args.zinc_padding_dim
        self.adjacency_matrix = torch.zeros((args.zinc_padding_dim, args.zinc_padding_dim))

    def integrality_check(self):
        for atom in self.atom_list:
            if atom is not None and atom.remain_valence() > 0:
                return False
        return True


# explicit H: 0, 1, 2, 3
class Atom:
    def __init__(self, list_):
        weight, valence = list_
        self.atom = Chem.Atom(weight)
        self.weight = weight
        self.max_valence = valence
        self.valence = 0

    def remain_valence(self):
        if self.max_valence - self.valence >= 0:
            return self.max_valence - self.valence
        else:
            raise AssertionError('Valence over maximum, max valence: ' + str(self.max_valence) + ' valence: ' + str(self.valence))

    def add_valence(self, valence):
        self.valence += valence
        self.remain_valence()


def renew_edge_probability(edge):
    return edge[:, :, :-1].max(dim=2)[0]


def argmax2d(tensor):
    shape = tensor.shape
    amax = tensor.argmax()
    return int(amax / shape[0]), int(amax % shape[0])


def ZINC_sync_recon(h_, edge_, atom_num=None):
    # h.shape: zinc_padding_dim, 15 or 6
    # edge.shape: zinc_padding_dim ** 2, 6
    # C | N | O | F
    # 4 | 3 | 2 | 1
    # single double triple none
    h_ = h_.cpu()
    edge_ = edge_.cpu()
    with torch.no_grad():
        if h_.shape == (args.zinc_padding_dim, 7 * (1 + args.charge_power) + 4):
            h_ = node_reprocess(h_[:, :7 * (1 + args.charge_power)])
        elif h_.shape == (args.zinc_padding_dim, 7 + 4):
            h_ = h_[:, :7]
        else:
            raise AssertionError(
                'Input shape error: the shape of h is ' + str(h_.shape[-1]))

        edge_ = edge_[:, :-1].reshape(args.zinc_padding_dim, args.zinc_padding_dim, 4)
        if atom_num is not None:
            edge_ = edge_[:atom_num, :atom_num, :]
            h_ = h_[:atom_num, :-1]  # h.shape -> (atom_num, 5)
        edge_ = (edge_ + edge_.transpose(0, 1)) / 2
        h_ = h_.argmax(dim=1)
        edge_ = edge_.argmax(dim=2)
        edge_ += 1
        mol = Chem.RWMol(Chem.MolFromSmiles(''))
        for atom in h_:
            if atom != 4:
                mol.AddAtom(ZINC_index_to_rdkit_atom(str(atom.numpy().tolist())))
        for bond_start in range(len(h_)):
            for bond_end in range(bond_start, len(h_)):
                if edge_[bond_start, bond_end] != 4:
                    mol.AddBond(bond_start, bond_end, num_to_bond_type(edge_[bond_start, bond_end]))
        return mol


def ZINC_async_recon(h_, edge_, bias=torch.ones(7).to(torch.float32), atom_num=None, debug=False):
    # h.shape: zinc_padding_dim, 30 or 10
    # edge.shape: zinc_padding_dim ** 2, 6
    # C | N | O | F
    # 4 | 3 | 2 | 1 # valence
    # 0 | 1 | 2 | 3 # index
    # single double triple none
    # start_time = time.time()
    h_ = h_.cpu()
    edge_ = edge_.cpu()

    if h_.shape == (args.zinc_padding_dim, 7 * (1 + args.charge_power) + 4):
        hs_ = h_[:, 7 * (1 + args.charge_power):]
        h_ = node_reprocess(h_[:, :7 * (1 + args.charge_power)].reshape(1, args.zinc_padding_dim, 7 * (1 + args.charge_power))).squeeze()
    elif h_.shape == (args.zinc_padding_dim, 7 + 4):
        hs_ = h_[:, 7:]
    else:
        raise AssertionError(
            'Input shape error: the shape of h is ' + str(h_.shape[-1]))
    h_ = f.softmax(h_, dim=1)
    h_ *= bias
    edge_ = edge_.reshape(args.zinc_padding_dim, args.zinc_padding_dim, -1)
    edge_ = (edge_ + edge_.transpose(0, 1)) / 2


    edge_ = edge_[:, :, :-1]
    for i in range(args.zinc_padding_dim):
        edge_[i, i, :] -= 10

    if atom_num is not None:
        edge_ = edge_[:atom_num, :atom_num]  # edge.shape -> (atom_num, atom_num, 3)
        h_ = h_[:atom_num, :-1]  # h.shape -> (atom_num, 10)

    mol = Molecule()

    for i in range(args.zinc_padding_dim):
        if h_[i].argmax() != 6:
            mol.atom_list[i] = ZINC_index_to_atom(h_[i].argmax(), hs=hs_[i].argmax())

    for i in range(args.zinc_padding_dim):
        if mol.atom_list[i] is not None:
            if mol.atom_list[i].remain_valence() == 0:
                raise AssertionError('Not connected graph')

    if debug:
        print(1)
    for i in range(args.zinc_padding_dim):
        if mol.atom_list[i] is not None:
            if mol.atom_list[i].remain_valence() == 1:
                edge_[i, :, 1:3] = 0
            elif mol.atom_list[i].remain_valence() == 2:
                edge_[i, :, 2] = 0

    if debug:
        print(2)

    edge_probability = renew_edge_probability(edge_)
    for i in range(args.zinc_padding_dim):
        if mol.atom_list[i] is not None and mol.atom_list[i].remain_valence() == 1:
            target_not_found = True
            loop_time = 0
            while target_not_found and loop_time < args.zinc_padding_dim:
                loop_time += 1
                sampled_edge_index = edge_probability[i].argmax()
                sampled_edge_type = 1
                edge_[i, sampled_edge_index, sampled_edge_type - 1] = 0
                edge_[sampled_edge_index, i, sampled_edge_type - 1] = 0
                if debug:
                    print(sampled_edge_index, i)
                    print(edge_probability[i])
                if mol.atom_list[sampled_edge_index] is not None and mol.atom_list[sampled_edge_index].remain_valence() >= sampled_edge_type and sampled_edge_index != i:
                    mol.atom_list[sampled_edge_index].add_valence(sampled_edge_type)
                    mol.atom_list[i].add_valence(sampled_edge_type)
                    mol.adjacency_matrix[i, sampled_edge_index] = sampled_edge_type
                    mol.adjacency_matrix[sampled_edge_index, i] = sampled_edge_type
                    target_not_found = False
                edge_[i, sampled_edge_index] = 0
                edge_[sampled_edge_index, i] = 0
                edge_probability = renew_edge_probability(edge_)

    if mol.integrality_check():
        return rdkit_mol(mol)

    if debug:
        print(3)
    for i in range(args.zinc_padding_dim):
        if mol.atom_list[i] is not None and mol.atom_list[i].remain_valence() == 1:
            target_not_found = True
            loop_time = 0
            while target_not_found and loop_time < 5:
                loop_time += 1
                sampled_edge_index = edge_probability[i].argmax()
                sampled_edge_type = 1
                edge_[i, sampled_edge_index, sampled_edge_type - 1] = 0
                edge_[sampled_edge_index, i, sampled_edge_type - 1] = 0
                if mol.atom_list[sampled_edge_index] is not None and mol.atom_list[sampled_edge_index].remain_valence() >= sampled_edge_type and sampled_edge_index != i:
                    mol.atom_list[sampled_edge_index].add_valence(sampled_edge_type)
                    mol.atom_list[i].add_valence(sampled_edge_type)
                    mol.adjacency_matrix[i, sampled_edge_index] = sampled_edge_type
                    mol.adjacency_matrix[sampled_edge_index, i] = sampled_edge_type
                    target_not_found = False
                edge_[i, sampled_edge_index] = 0
                edge_[sampled_edge_index, i] = 0
                edge_probability = renew_edge_probability(edge_)

    if mol.integrality_check():
        return rdkit_mol(mol)

    if debug:
        print(4)
    for i in range(args.zinc_padding_dim):
        if mol.atom_list[i] is not None and mol.adjacency_matrix[i].sum() == 0:
            target_not_found = True
            loop_time = 0
            while target_not_found and loop_time < 3:
                loop_time += 1
                sampled_edge_index = edge_probability[i].argmax()
                sampled_edge_type = edge_[i, sampled_edge_index, :-1].argmax() + 1
                edge_[i, sampled_edge_index, sampled_edge_type - 1] = 0
                edge_[sampled_edge_index, i, sampled_edge_type - 1] = 0
                if debug:
                    print(4.1)
                if mol.atom_list[sampled_edge_index] is not None and mol.atom_list[i].remain_valence(
                ) >= sampled_edge_type and mol.atom_list[sampled_edge_index].remain_valence() >= sampled_edge_type:
                    if debug:
                        print(sampled_edge_type)
                    mol.atom_list[sampled_edge_index].add_valence(sampled_edge_type)
                    mol.atom_list[i].add_valence(sampled_edge_type)
                    mol.adjacency_matrix[i, sampled_edge_index] = sampled_edge_type
                    mol.adjacency_matrix[sampled_edge_index, i] = sampled_edge_type
                    target_not_found = False
                edge_[i, sampled_edge_index] = 0
                edge_[sampled_edge_index, i] = 0
                edge_probability = renew_edge_probability(edge_)

    if mol.integrality_check():
        return rdkit_mol(mol)

    big_loop = 0
    while not mol.integrality_check() and big_loop < 10:
        big_loop += 1
        if debug:
            print(5)
        for i in range(args.zinc_padding_dim):
            if mol.atom_list[i] is not None and mol.atom_list[i].remain_valence() > 0:
                target_not_found = True
                loop_time = 0
                while target_not_found and loop_time < 3:
                    loop_time += 1
                    if debug:
                        print(5.1)
                    sampled_edge_index = edge_probability[i].argmax()
                    sampled_edge_type = edge_[i, sampled_edge_index, :-1].argmax() + 1
                    edge_[i, sampled_edge_index, sampled_edge_type - 1] = 0
                    edge_[sampled_edge_index, i, sampled_edge_type - 1] = 0
                    if mol.atom_list[sampled_edge_index] is not None and mol.atom_list[i].remain_valence(

                    ) >= sampled_edge_type and mol.atom_list[sampled_edge_index].remain_valence(

                    ) >= sampled_edge_type and i != sampled_edge_index:
                        if debug:
                            print(mol.atom_list[sampled_edge_index].remain_valence())
                            print(sampled_edge_type)
                        mol.atom_list[sampled_edge_index].add_valence(sampled_edge_type)
                        mol.atom_list[i].add_valence(sampled_edge_type)
                        mol.adjacency_matrix[i, sampled_edge_index] = sampled_edge_type
                        mol.adjacency_matrix[sampled_edge_index, i] = sampled_edge_type
                        target_not_found = False
                    edge_[i, sampled_edge_index] = 0
                    edge_[sampled_edge_index, i] = 0
                    edge_probability = renew_edge_probability(edge_)

    return rdkit_mol(mol)


if __name__ == '__main__':
    # 'C#C/C(C)=N/O'

    data = np.load('DataU_DebugData/h_origin.npz')
    h = torch.tensor(data['arr_0'])
    hs = torch.tensor(data['arr_1'])
    edge = torch.tensor(data['arr_2'])
    # m = QM_sync_recon(torch.cat((h[0], hs[0]), dim=1), edge[0])
    # m = QM_async_recon(torch.cat((h[0], hs[0]), dim=1), edge[0], atom_num=None, debug=True)
    # print(Chem.MolToSmiles(m))
    pass
