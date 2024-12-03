# import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from A_DataPreperation.AU_util import *
from B_Train.BU_util import *
from U_Model.U_AREVGA import AREVGA
from U_Model.U_util import *
from args import args
import os

device = args.device
dtype = torch.float32
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
LAST_EPOCH = 1


def general_edge(padding_dim_):
    edges = torch.zeros((args.batch_size, 2, padding_dim_ ** 2))
    for i_ in range(args.batch_size):
        for j_ in range(padding_dim_):
            for k_ in range(padding_dim_):
                edges[i_][0][j_ * padding_dim_ + k_] = j_
                edges[i_][1][j_ * padding_dim_ + k_] = k_
    edges = edge_process(edges, padding_dim_)
    return edges


def train_AREVGA_ZINC(retrain=True, attention=True):
    padding_dim = args.zinc_padding_dim
    if attention:
        encoder_file = 'DataC_SavedModel/AREVGA_ZINC_attention_encoder.model'
        decoder_file = 'DataC_SavedModel/AREVGA_ZINC_attention_decoder.model'
        cri_file = 'DataC_SavedModel/AREVGA_ZINC_attention_cri.model'
        gen_file = 'DataC_SavedModel/AREVGA_ZINC_attention_gen.model'
        pre_file = 'DataC_SavedModel/AREVGA_ZINC_attention_pre.model'
    else:
        encoder_file = 'DataC_SavedModel/AREVGA_ZINC_encoder.model'
        decoder_file = 'DataC_SavedModel/AREVGA_ZINC_decoder.model'
        cri_file = 'DataC_SavedModel/AREVGA_ZINC_cri.model'
        gen_file = 'DataC_SavedModel/AREVGA_ZINC_gen.model'
        pre_file = 'DataC_SavedModel/AREVGA_ZINC_pre.model'

    model = AREVGA(in_node_nf=7 * (args.charge_power + 1) + 4, in_edge_nf=5, hidden_nf=args.hidden_feature, latent_dim=args.latent_dim,
                   device=device, n_layers=args.n_layers, attention=attention, node_attr=args.node_attr, padding_dim=args.zinc_padding_dim)

    zinc_dataset = ZINC_dataset_()
    pre_dataset = Pre_train_dataset_()
    pre_dataloader = DataLoader(pre_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    zinc_train_dataset, zinc_valid_dataset, zinc_test_dataset = random_split(
        zinc_dataset, lengths=[int(len(zinc_dataset) * 0.7),
                               int(len(zinc_dataset) * 0.2),
                               len(zinc_dataset) - int(len(zinc_dataset) * 0.7) - int(len(zinc_dataset) * 0.2)],
        generator=torch.Generator().manual_seed(1215))
    zinc_train_dataloader = DataLoader(zinc_train_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, drop_last=True)
    zinc_valid_dataloader = DataLoader(zinc_valid_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, drop_last=True)
    zinc_test_dataloader = DataLoader(zinc_test_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, drop_last=True)

    optimizer_AE = optim.Adam(
        [{'params': list(model.encoder.parameters()) + list(model.decoder.parameters()), 'initial_lr': args.aelr}], weight_decay=args.weight_decay)
    optimizer_gen = optim.Adam(
        [{'params': list(model.encoder.parameters()) + list(model.generator.parameters()), 'initial_lr': args.genlr}], weight_decay=args.weight_decay)
    optimizer_cri = optim.Adam(
        [{'params': model.critic.parameters(), 'initial_lr': args.crilr}], weight_decay=args.weight_decay)

    criterion_AE_h = nn.L1Loss()
    criterion_AE_coord = nn.L1Loss()
    criterion_AE_edge_attr = nn.L1Loss()
    criterion_AE_atom_num = nn.CrossEntropyLoss()

    lr_scheduler_AE = optim.lr_scheduler.StepLR(
        optimizer_AE, step_size=4, gamma=0.8, last_epoch=LAST_EPOCH)
    lr_scheduler_gen = optim.lr_scheduler.StepLR(
        optimizer_gen, step_size=4, gamma=0.8, last_epoch=LAST_EPOCH)
    lr_scheduler_cri = optim.lr_scheduler.StepLR(
        optimizer_cri, step_size=4, gamma=0.8, last_epoch=LAST_EPOCH)


    if retrain:
        print('Reloading trained model')
        model.encoder.load_state_dict(torch.load(encoder_file))
        model.decoder.load_state_dict(torch.load(decoder_file))
        model.critic.load_state_dict(torch.load(cri_file))
        model.generator.load_state_dict(torch.load(gen_file))

    def train(epoch_, loader, partition='train'):
        res_ae = {'loss': 0, 'counter': 1, 'loss_arr': []}
        res_gen = {'loss': 0, 'counter': 1, 'loss_arr': []}
        res_cri = {'loss': 0, 'counter': 1, 'loss_arr': []}
        for i___, data in enumerate(loader):
            edges = general_edge(padding_dim)
            if partition != 'train' and i___ == 1000:
                break
            batch_size, n_nodes, _ = data['position_list'].size()
            if partition == 'train':
                model.train()
            else:
                model.eval()
            atom_positions = data['position_list'].view(batch_size * n_nodes, -1).to(device)
            one_hot = data['charge_onehot_list'].to(device)
            charges = data['charge_list'].to(device)
            # atom_num = data['num_atom_list'].to(device)
            nodes = node_process(one_hot, charges).to(device)
            hs = data['explicit_h_list'].to(device)
            nodes = nodes.view(batch_size * n_nodes, -1)
            hs = hs.view(batch_size * n_nodes, -1)
            score = data['score_list'].to(device)
            score = torch.cat([score] * args.zinc_padding_dim, dim=1)
            nodes = torch.cat((nodes, hs), dim=1)
            edge_feature = data['edge_feature_list'].view(-1, 5).to(device) * 0.25
            h_re, coord_re, edge_attr_re, atom_num_re = model(h0=nodes,
                                                              label=score.view(batch_size * n_nodes, -1),
                                                              x=atom_positions,
                                                              edges=edges,
                                                              edge_attr=edge_feature,
                                                              n_nodes=n_nodes)
            loss_AE = criterion_AE_h(
                h_re, nodes) + criterion_AE_edge_attr(
                edge_attr_re[:, :-1], edge_feature[:, :-1]
            )  # + criterion_AE_atom_num(atom_num_re, (atom_num - 1).squeeze().long()) + criterion_AE_coord(coord_re, atom_positions)
            prefix = ""
            if partition == 'train':
                optimizer_AE.zero_grad()
                loss_AE.backward()
                # nn.utils.clip_grad_norm_(AE_params, max_norm=2000., norm_type=2)
                optimizer_AE.step()
            else:
                prefix = ">> %s \t" % partition
            res_ae['loss'] += loss_AE.item() * batch_size
            res_ae['counter'] += batch_size
            res_ae['loss_arr'].append(loss_AE.item())

            z_real = model.encoder.encode(
                h0=nodes, label=score.view(batch_size * n_nodes, -1), x=atom_positions, edges=edges, edge_attr=edge_feature, n_nodes=n_nodes)
            batch_s = torch.normal(mean=torch.zeros(batch_size, padding_dim * args.latent_dim), std=1.).to(device, dtype)
            z_gen = model.generator(batch_s)
            d_gen = model.critic(z_gen)
            d_real = model.critic(z_real.view(batch_size, -1))
            optimizer_cri.zero_grad()
            loss_cri = - d_real.mean() + d_gen.mean()
            if partition == 'train':
                loss_cri.backward()
                optimizer_cri.step()
            res_cri['loss'] += loss_cri.item() * batch_size / args.crin / args.genn
            res_cri['counter'] += batch_size / args.crin / args.genn
            batch_s = torch.normal(mean=torch.zeros(batch_size, padding_dim * args.latent_dim), std=1.).to(device, dtype)
            z_real = model.encoder.encode(
                h0=nodes, label=score.view(batch_size * n_nodes, -1), x=atom_positions, edges=edges, edge_attr=edge_feature, n_nodes=n_nodes)
            d_real = model.critic(z_real.view(batch_size, -1))
            z_gen = model.generator(batch_s)
            d_gen = model.critic(z_gen)
            optimizer_gen.zero_grad()
            loss_gen = - d_gen.mean() + d_real.mean()
            if partition == 'train':
                loss_gen.backward()
                optimizer_gen.step()
            res_gen['loss'] += loss_gen.item() * batch_size / args.genn
            res_gen['counter'] += batch_size / args.genn
            if i___ % args.log_interval == 0:
                print(prefix + "\033[0;31;40mEpoch\033[0m %d \t \033[0;32;40mIteration\033[0m %d \t \033[0;33;40mAE loss\033[0m %.4f" % (
                    epoch_, i___, sum(res_ae['loss_arr'][-50:]) / len(res_ae['loss_arr'][-50:]) * 1000))
                if partition == 'train':
                    lr_scheduler_AE.step()
                    lr_scheduler_cri.step()
                    lr_scheduler_gen.step()
            '''
            lr_scheduler_AE.step(res_ae['loss'] / res_ae['counter'])
            lr_scheduler_cri.step(res_cri['loss'] / res_cri['counter'])
            lr_scheduler_gen.step(res_gen['loss'] / res_gen['counter'])
            '''

        return [(res_ae['loss'] / res_ae['counter']),
                (res_gen['loss'] / res_gen['counter']),
                (res_cri['loss'] / res_cri['counter'])]

    res_ae = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}
    res_gen = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}
    res_cri = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}

    train(0, pre_dataloader, 'train')

    for epoch in range(args.epochs):
        train(epoch, zinc_train_dataloader, 'train')
        train(epoch, pre_dataloader, 'train')
        if epoch % args.test_interval == 0:
            val_loss = train(epoch, zinc_valid_dataloader, partition='valid')
            test_loss = train(epoch, zinc_test_dataloader, partition='test')
            res_ae['epochs'].append(epoch)
            res_gen['epochs'].append(epoch)
            res_cri['epochs'].append(epoch)
            res_ae['losess'].append(test_loss[0])
            res_gen['losess'].append(test_loss[1])
            res_cri['losess'].append(test_loss[2])

            if val_loss[0] < res_ae['best_val'] or epoch == 0:
                res_ae['best_val'] = val_loss[0]
                res_ae['best_test'] = test_loss[0]
                res_ae['best_epoch'] = epoch
                torch.save(model.encoder.state_dict(), encoder_file)
                torch.save(model.decoder.state_dict(), decoder_file)
                torch.save(model.generator.state_dict(), gen_file)
                torch.save(model.critic.state_dict(), cri_file)
                print('Model Saved')


if __name__ == '__main__':
    # train_AREVGA_QM9(retrain=False, attention=True)
    train_AREVGA_ZINC(retrain=False, attention=True)
