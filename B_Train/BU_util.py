import torch
from args import args

charge_dict = {'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17}
CHARGE_LIST = [6, 7, 8, 9, 15, 16]


def node_process(one_hot, charges, charge_power=args.charge_power, charge_scale=args.charge_scale):
    # charge_tensor = torch.ones_like(charges.unsqueeze(-1)).pow(torch.arange(charge_power + 1., dtype=torch.float32).to(args.device))
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(torch.arange(charge_power + 1., dtype=torch.float32).to(args.device))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    atom_scalars[:, :, -4] = atom_scalars[:, :, -5] * 0.35
    atom_scalars[:, :, -3] = atom_scalars[:, :, -5] * 0.15
    atom_scalars[:, :, -2] = atom_scalars[:, :, -5] * 0.1
    atom_scalars[:, :, -1] = atom_scalars[:, :, -5] * 0.02
    return atom_scalars


def node_reprocess(atom_scalars, charge_scale=args.charge_scale):  # TODO 把这个改了，在模型训练好之前
    atom_scalars = torch.clamp(atom_scalars, min=0, max=1)
    charge_power = args.charge_power  # 2
    node_re = torch.zeros_like(atom_scalars)[:, :, :7]
    for i in range(6):
        atom_scalars[:, :, i * (charge_power + 1)] = (
            atom_scalars[:, :, i * (charge_power + 1)] + atom_scalars[:, :, i * (charge_power + 1) + 1] * charge_scale / CHARGE_LIST[i] + (
                                                         atom_scalars[:, :, i * (charge_power + 1) + 2]) ** (1 / 2) * charge_scale / CHARGE_LIST[i]
                                                     ) / 3
        for j in range(args.charge_power + 1):
            node_re[:, :, i] += (atom_scalars[:, :, i * (charge_power + 1) + j]) ** (1 / (j + 1)) * charge_scale / CHARGE_LIST[i] / (charge_power + 1)
    node_re[:, :, 6] = torch.sum(atom_scalars[:, :, -(args.charge_power + 1):], dim=2)
    return node_re / (charge_power + 1)
    # return atom_scalars[:, :, [0, 3, 6, 9, 12, 15, 18]]


def edge_process(edge_batch, padding_dim_):
    for i in range(edge_batch.shape[0]):
        for j in range(edge_batch.shape[2]):
            edge_batch[i, 0, j] += i * padding_dim_
            edge_batch[i, 1, j] += i * padding_dim_
    return [edge_batch[:, 0, :].reshape(-1).long(), edge_batch[:, 1, :].reshape(-1).long()]


def mean_and_mad(property_):
    mean_ = torch.mean(property_, dim=0)
    mad_ = torch.mean(torch.abs(property_ - mean_), dim=0)
    return mean_, mad_
