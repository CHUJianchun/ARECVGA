from torch.utils.data import DataLoader, random_split
from tqdm import tqdm, trange
import numpy as np

from A_DataPreperation.AU_util import QM_dataset_, ZINC_dataset_
from B_Train.BU_util import *
from U_Model.U_AREVGA import AREVGA
from U_Model.U_util import *
from args import args


def get_latent_code(attention=True):  # need HUGE memory!!!
    device = args.device
    dtype = torch.float32
    print('Loading model...')
    if attention:
        model = AREVGA(in_node_nf=30, in_edge_nf=5, hidden_nf=args.hidden_feature, latent_dim=args.latent_dim,
                       device=device, n_layers=args.n_layers, attention=attention, node_attr=args.node_attr)
        model.encoder.load_state_dict(torch.load('DataC_SavedModel/AREVGA_attention_encoder.model'))
        model.decoder.load_state_dict(torch.load('DataC_SavedModel/AREVGA_attention_decoder.model'))
        model.critic.load_state_dict(torch.load('DataC_SavedModel/AREVGA_attention_cri.model'))
        model.generator.load_state_dict(torch.load('DataC_SavedModel/AREVGA_attention_gen.model'))
        model.property_mlp.load_state_dict(torch.load('DataC_SavedModel/AREVGA_attention_pre.model'))
    else:
        model = AREVGA(in_node_nf=18, in_edge_nf=5, hidden_nf=args.hidden_feature, latent_dim=args.latent_dim,
                       device=device, n_layers=args.n_layers, attention=False, node_attr=args.node_attr)
        model.encoder.load_state_dict(torch.load('DataC_SavedModel/AREVGA_encoder.model'))
        model.decoder.load_state_dict(torch.load('DataC_SavedModel/AREVGA_decoder.model'))
        model.critic.load_state_dict(torch.load('DataC_SavedModel/AREVGA_cri.model'))
        model.generator.load_state_dict(torch.load('DataC_SavedModel/AREVGA_gen.model'))
        model.property_mlp.load_state_dict(torch.load('DataC_SavedModel/AREVGA_pre.model'))

    dataset_list = [QM_dataset_]
    for i in range(8):
        dataset_list.append(ZINC_dataset_)
    latent_code_list = []
    model.eval()
    with torch.no_grad():
        edges = torch.zeros((1, 2, args.padding_dim ** 2))
        for j_ in range(args.padding_dim):
            for k_ in range(args.padding_dim):
                edges[0][0][j_ * args.padding_dim + k_] = j_
                edges[0][1][j_ * args.padding_dim + k_] = k_
        edges = edge_process(edges)
        for i in range(len(dataset_list)):
            dataset = dataset_list[i]() if i == 0 else dataset_list[i](i - 1)
            for i__ in range(dataset.data['adjacency_list'].shape[0]):
                atom_num = dataset.data['num_atom_list'][i__].int().numpy()[0]
                dataset.data['adjacency_list'][i__] += torch.eye(args.padding_dim)
                dataset.data['adjacency_list'][i__, atom_num:, atom_num:] = 1.
            for i__ in trange(len(dataset)):
                data = dataset[i__]
                n_nodes, _ = data['position_list'].size()
                atom_positions = data['position_list'].view(n_nodes, -1).to(device, dtype)
                one_hot = data['charge_onehot_list'].to(device, dtype)
                charges = data['charge_list'].to(device, dtype)
                nodes = node_process(one_hot, charges)
                nodes = nodes.view(n_nodes, -1).to(device, dtype)
                edge_feature = data['edge_feature_list'].view(-1, 5).to(device, dtype)
                z = model.encoder.encode(h0=nodes, x=atom_positions, edges=edges, edge_attr=edge_feature, n_nodes=n_nodes).detach().cpu().numpy()
                latent_code_list.append(z)
    latent_code_list = np.array(latent_code_list)
    np.save('DataB_ProcessedData/All_Mol_latent_code.npy', latent_code_list)


def write_latent_code(attention=True):  # need HUGE memory!!!
    device = args.device
    dtype = torch.float32
    print('Loading model...')
    if attention:
        model = AREVGA(in_node_nf=30, in_edge_nf=5, hidden_nf=args.hidden_feature, latent_dim=args.latent_dim,
                       device=device, n_layers=args.n_layers, attention=attention, node_attr=args.node_attr)
        model.encoder.load_state_dict(torch.load('DataC_SavedModel/AREVGA_attention_encoder.model'))
        model.decoder.load_state_dict(torch.load('DataC_SavedModel/AREVGA_attention_decoder.model'))
        model.critic.load_state_dict(torch.load('DataC_SavedModel/AREVGA_attention_cri.model'))
        model.generator.load_state_dict(torch.load('DataC_SavedModel/AREVGA_attention_gen.model'))
        model.property_mlp.load_state_dict(torch.load('DataC_SavedModel/AREVGA_attention_pre.model'))
    else:
        model = AREVGA(in_node_nf=18, in_edge_nf=5, hidden_nf=args.hidden_feature, latent_dim=args.latent_dim,
                       device=device, n_layers=args.n_layers, attention=False, node_attr=args.node_attr)
        model.encoder.load_state_dict(torch.load('DataC_SavedModel/AREVGA_encoder.model'))
        model.decoder.load_state_dict(torch.load('DataC_SavedModel/AREVGA_decoder.model'))
        model.critic.load_state_dict(torch.load('DataC_SavedModel/AREVGA_cri.model'))
        model.generator.load_state_dict(torch.load('DataC_SavedModel/AREVGA_gen.model'))
        model.property_mlp.load_state_dict(torch.load('DataC_SavedModel/AREVGA_pre.model'))

    dataset_list = [QM_dataset_]
    for i in range(8):
        dataset_list.append(ZINC_dataset_)
    model.eval()
    with torch.no_grad():
        edges = torch.zeros((1, 2, args.padding_dim ** 2))
        for j_ in range(args.padding_dim):
            for k_ in range(args.padding_dim):
                edges[0][0][j_ * args.padding_dim + k_] = j_
                edges[0][1][j_ * args.padding_dim + k_] = k_
        edges = edge_process(edges)
        for i in range(len(dataset_list)):
            if i > 0:  # TODO delete
                continue
            latent_code_list = []
            dataset, path = dataset_list[i](return_path=True) if i == 0 else dataset_list[i](i - 1, return_path=True)
            for i__ in range(dataset.data['adjacency_list'].shape[0]):
                atom_num = dataset.data['num_atom_list'][i__].int().numpy()[0]
                dataset.data['adjacency_list'][i__] += torch.eye(args.padding_dim)
                dataset.data['adjacency_list'][i__, atom_num:, atom_num:] = 1.
            for i__ in trange(len(dataset)):
                data = dataset[i__]
                n_nodes, _ = data['position_list'].size()
                atom_positions = data['position_list'].view(n_nodes, -1).to(device, dtype)
                one_hot = data['charge_onehot_list'].to(device, dtype)
                charges = data['charge_list'].to(device, dtype)
                nodes = node_process(one_hot, charges)
                nodes = nodes.view(n_nodes, -1).to(device, dtype)
                edge_feature = data['edge_feature_list'].view(-1, 5).to(device, dtype)
                z = model.encoder.encode(h0=nodes, x=atom_positions, edges=edges, edge_attr=edge_feature, n_nodes=n_nodes).detach().cpu().numpy()
                latent_code_list.append(z)
            latent_code_list = np.array(latent_code_list)
            if i == 0:
                np.savez(path,
                         smiles_list=dataset.data['smiles_list'],
                         charge_list=dataset.data['charge_list'],
                         adjacency_list=dataset.data['adjacency_list'],
                         num_atom_list=dataset.data['num_atom_list'],
                         position_list=dataset.data['position_list'],
                         property_list=dataset.data['property_list'],
                         charge_onehot_list=dataset.data['charge_onehot_list'],
                         edge_feature_list=dataset.data['edge_feature_list'],
                         latent_code_list=latent_code_list)
            else:
                np.savez(path,
                         smiles_list=dataset.data['smiles_list'],
                         charge_list=dataset.data['charge_list'],
                         adjacency_list=dataset.data['adjacency_list'],
                         num_atom_list=dataset.data['num_atom_list'],
                         position_list=dataset.data['position_list'],
                         charge_onehot_list=dataset.data['charge_onehot_list'],
                         edge_feature_list=dataset.data['edge_feature_list'],
                         latent_code_list=latent_code_list)
    np.save('DataB_ProcessedData/All_Mol_latent_code.npy', latent_code_list)


if __name__ == '__main__':
    # get_latent_code()
    write_latent_code()
