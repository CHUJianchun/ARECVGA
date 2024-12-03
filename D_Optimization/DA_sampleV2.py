from tqdm import tqdm, trange
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import torch
from torch import tensor, optim
from A_DataPreperation.AU_util import ELE_dataset_
from B_Train.BU_util import *
from U_Model.U_AREVGA import AREVGA
from U_Model.U_util import *
from U_reconstruct.U_reconstruct_methods import ZINC_sync_recon, ZINC_async_recon
from args import args
import os
import shutil

device = args.device
dtype = torch.float32
scale = 0.25


def random_search(target, result_num=500, attention=True, bias=torch.ones(7).to(torch.float32), decode_mode='async',
                  burte_force_link=True, debug=False):
    assert len(target) == 7

    def save_path_():
        folder = 'DataE_Figure/Sample_' + '_'.join(map(str, target))
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            shutil.rmtree(folder)
            os.makedirs(folder)
        return folder + '/'

    save_path = save_path_()
    target = torch.tensor(target).to(device, dtype)
    target /= 5
    target = target.repeat(args.batch_size * args.zinc_padding_dim, 1)
    mol_list = []
    path = 'DataC_SavedModel/'

    if attention:
        model = AREVGA(in_node_nf=7 * (args.charge_power + 1) + 4, in_edge_nf=5, hidden_nf=args.hidden_feature, latent_dim=args.latent_dim,
                       device=device, n_layers=args.n_layers, attention=attention, node_attr=args.node_attr, padding_dim=args.zinc_padding_dim)
        model.encoder.load_state_dict(torch.load(path + 'AREVGA_ZINC_attention_encoder.model'))
        model.decoder.load_state_dict(torch.load(path + 'AREVGA_ZINC_attention_decoder.model'))
        model.critic.load_state_dict(torch.load(path + 'AREVGA_ZINC_attention_cri.model'))
        model.generator.load_state_dict(torch.load(path + 'AREVGA_ZINC_attention_gen.model'))
        # model.property_mlp.load_state_dict(torch.load('DataC_SavedModel/AREVGA_ZINC_attention_pre.model'))
    else:
        model = AREVGA(in_node_nf=7 * (args.charge_power + 1) + 4, in_edge_nf=5, hidden_nf=args.hidden_feature, latent_dim=args.latent_dim,
                       device=device, n_layers=args.n_layers, attention=attention, node_attr=args.node_attr, padding_dim=args.zinc_padding_dim)
        model.encoder.load_state_dict(torch.load(path + 'AREVGA_ZINC_encoder.model'))
        model.decoder.load_state_dict(torch.load(path + 'AREVGA_ZINC_decoder.model'))
        model.critic.load_state_dict(torch.load(path + 'AREVGA_ZINC_cri.model'))
        model.generator.load_state_dict(torch.load(path + 'AREVGA_ZINC_gen.model'))
    model.eval()
    with torch.no_grad():
        edges = torch.zeros((args.batch_size, 2, args.zinc_padding_dim ** 2))
        for i_ in range(args.batch_size):
            for j_ in range(args.zinc_padding_dim):
                for k_ in range(args.zinc_padding_dim):
                    edges[i_][0][j_ * args.zinc_padding_dim + k_] = j_
                    edges[i_][1][j_ * args.zinc_padding_dim + k_] = k_
        edges = edge_process(edges, args.zinc_padding_dim)
        rand_latent = (torch.rand(result_num, args.batch_size * args.zinc_padding_dim, args.latent_dim) * scale).to(device)
        # 180, 24 (batch_size * padding_dim, latent_dim)
        for i in range(result_num):
            h_re, coord_re, edge_attr_re, atom_num_re = model.decoder(rand_latent[i], label=target, n_nodes=args.zinc_padding_dim, edges=edges)
            h_re = h_re.reshape(args.batch_size, args.zinc_padding_dim, -1)
            edge_attr_re = edge_attr_re.reshape(args.batch_size, args.zinc_padding_dim, args.zinc_padding_dim, 5)
            if decode_mode == 'async':
                for i_ in trange(h_re.shape[0]):
                    mol = ZINC_async_recon(h_re[i_], edge_attr_re[i_], bias=bias, debug=debug)
                    if burte_force_link:
                        try:
                            smiles = Chem.MolToSmiles(mol).replace('.', '')
                            # smiles = ''.join([i for i in smiles if not i.isdigit() or int(i) <= 2])
                            mol = Chem.MolFromSmiles(smiles)
                            graph_file_name = save_path + str(i) + '_' + Chem.MolToSmiles(mol) + '_.png'
                            Draw.MolToFile(mol, graph_file_name)
                        except:
                            # mol = Chem.MolFromSmiles('C')
                            # graph_file_name = save_path + str(i) + '_' + Chem.MolToSmiles(mol) + '_.png'
                            # Draw.MolToFile(mol, graph_file_name)
                            pass
                    else:
                        try:
                            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
                            graph_file_name = save_path + str(i) + '_' + Chem.MolToSmiles(mol) + '_.png'
                            Draw.MolToFile(mol, graph_file_name)
                        except:
                            # mol = Chem.MolFromSmiles('C')
                            # graph_file_name = save_path + str(i) + '_' + Chem.MolToSmiles(mol) + '_.png'
                            # Draw.MolToFile(mol, graph_file_name)
                            pass
            elif decode_mode == 'sync':
                pass


# NOTE C N O F P S X
if __name__ == '__main__':
    # bias = torch.tensor([1, 1, 1, 1, 1, 1, 1]).to(dtype)
    bias = torch.tensor([1, 0.95, 0.9, 1.27, 1.15, 1.2, 1]).to(dtype)
    # random_search([5, 2, 5, 3.5, 4, 5, 1], bias=bias, debug=True)
    # random_search([5, 2, 5, 3.5, 4, 5, 1], bias=bias, debug=False)
    # random_search([5, 2.5, 5, 3.5, 4.5, 5, 1.5], bias=bias, debug=False)
    # random_search([5, 2, 5, 3.5, 4, 5, 1], debug=True)
