from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem import Draw

import torch
from torch import tensor, optim
from torch.utils.data import DataLoader

from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

from A_DataPreperation.AU_util import SmilesToData, ZINC_dataset_
from B_Train.BU_util import *
from U_Model.U_AREVGA import AREVGA
from U_Model.U_util import *
from U_reconstruct.U_reconstruct_methods import Sync_recon, Async_recon
from args import args


device = args.device
dtype = torch.float32
EPOCH = 3
choice = 25000
# TODO isomap
# NOTE EC -> FEC, .EP.
# NOTE TEP -> .EP., FTEP
# NOTE FEC, EP, FTEP -> TFEP

# NOTE EC:  C1COC(=O)O1
# NOTE TEP: O=P(OCC)(OCC)OCC
# NOTE FEC: C1C(F)OC(=O)O1
# NOTE EP: C1COP(=O)(OCC)O1
# NOTE FTEP: O=P(OCC(F)(F)F)(OCC(F)(F)F)OCC(F)(F)F
# NOTE TFEP: C1COP(=O)(OCC(F)(F)F)O1

smiles_list = ['C1COC(=O)O1', 'O=P(OCC)(OCC)OCC', 'C1C(F)OC(=O)O1',
               'C1COP(=O)(OCC)O1', 'O=P(OCC(F)(F)F)(OCC(F)(F)F)OCC(F)(F)F', 'C1COP(=O)(OCC(F)(F)F)O1']
attention = True


data = SmilesToData(smiles_list)
for key in data:
    data[key] = torch.tensor(data[key])
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
model.eval()

with torch.no_grad():
    batch_size, n_nodes, _ = data['position_list'].shape
    edges = torch.zeros((batch_size, 2, args.padding_dim ** 2))
    for i_ in range(batch_size):
        for j_ in range(args.padding_dim):
            for k_ in range(args.padding_dim):
                edges[i_][0][j_ * args.padding_dim + k_] = j_
                edges[i_][1][j_ * args.padding_dim + k_] = k_
    edges = edge_process(edges)
    atom_positions = data['position_list'].view(batch_size * n_nodes, -1).to(device, dtype)
    one_hot = data['charge_onehot_list'].to(device, dtype)
    charges = data['charge_list'].to(device, dtype)
    nodes = node_process(one_hot, charges)
    nodes = nodes.view(batch_size * n_nodes, -1).to(device, dtype)
    edge_feature = data['edge_feature_list'].view(-1, 5).to(device, dtype)
    evolution_z = model.encoder.encode(h0=nodes, x=atom_positions, edges=edges, edge_attr=edge_feature, n_nodes=n_nodes).reshape(
        batch_size, -1).detach().cpu().numpy()

    zinc_latent_code = []
    batch_size = 1000
    edges = torch.zeros((batch_size, 2, args.padding_dim ** 2))
    for i_ in range(batch_size):
        for j_ in range(args.padding_dim):
            for k_ in range(args.padding_dim):
                edges[i_][0][j_ * args.padding_dim + k_] = j_
                edges[i_][1][j_ * args.padding_dim + k_] = k_
    edges = edge_process(edges)
    for i in range(8):
        zinc_dataset = ZINC_dataset_(i)
        zinc_dataloader = DataLoader(zinc_dataset, batch_size=batch_size, drop_last=True)
        for j, data in tqdm(enumerate(zinc_dataloader)):
            batch_size, n_nodes, _ = data['position_list'].size()
            atom_positions = data['position_list'].view(batch_size * n_nodes, -1).to(device)
            one_hot = data['charge_onehot_list'].to(device)
            charges = data['charge_list'].to(device)
            nodes = node_process(one_hot, charges).to(device)
            nodes = nodes.view(batch_size * n_nodes, -1)
            edge_feature = data['edge_feature_list'].view(-1, 5).to(device)
            z_batch = model.encoder.encode(h0=nodes, x=atom_positions, edges=edges, edge_attr=edge_feature, n_nodes=n_nodes).reshape(
                batch_size, -1).detach().cpu().numpy()
            zinc_latent_code.append(z_batch)
    zinc_latent_code = np.array(zinc_latent_code)
    zinc_latent_code = zinc_latent_code.reshape(zinc_latent_code.shape[0] * zinc_latent_code.shape[1], -1)
np.save('DataD_OutputData/zinc_latent_code.npy', zinc_latent_code)
np.save('DataD_OutputData/electrolyte_latent_code.npy', evolution_z)
zinc_latent_code = np.load('DataD_OutputData/zinc_latent_code.npy')
evolution_z = np.load('DataD_OutputData/electrolyte_latent_code.npy')
evolution_z_simple = np.delete(evolution_z, -2, axis=0)
'''
zinc_latent_code_copy = np.copy(zinc_latent_code)
np.random.shuffle(zinc_latent_code_copy)
zinc_latent_code_choice = zinc_latent_code_copy[:choice]
embedding = Isomap(n_components=2, n_jobs=-1)
embedding.fit(np.concatenate((zinc_latent_code_copy, evolution_z)))
evolution_transformed = embedding.transform(evolution_z)
'''
embedding = PCA(n_components=2)
# evolution_transformed = embedding.fit_transform(evolution_z)
evolution_simple_transformed = embedding.fit_transform(evolution_z_simple)

embedding = PCA(n_components=3)
# evolution_transformed = embedding.fit_transform(evolution_z)
evolution_transformed = embedding.fit_transform(evolution_z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = evolution_transformed[:, 0]
ys = evolution_transformed[:, 1]
zs = evolution_transformed[:, 2]
ax.scatter(xs, ys, zs, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show(block=True)