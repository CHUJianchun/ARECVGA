import torch
from torch import nn
from U_Model.U_util import *
from U_Model.U_Layer import E_GCL_mask
from U_Model.U_MLP import MLP
from B_Train.BU_util import *
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


def reparameterize(mu_, log_var_):
    std = torch.exp(log_var_ / 2)
    eps = torch.randn_like(std)
    return mu_ + 0.01 * eps * std


class Encoder(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, latent_dim, device='cpu', act_fn=nn.ELU(), n_layers=2,
                 coords_weight=1.0, attention=False, node_attr=1, padding_dim=args.qm_padding_dim):
        super(Encoder, self).__init__()
        self.padding_dim = padding_dim
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        # Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = hidden_nf
        else:
            n_node_attr = 0

        for i in range(0, n_layers):
            self.add_module("encoder_%d" % i,
                            E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                       nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=False,
                                       coords_weight=coords_weight, attention=attention))

        self.g2z_mu = nn.Sequential(nn.Linear(self.hidden_nf + 7, self.latent_dim))

        self.g2z_var = nn.Sequential(nn.Linear(self.hidden_nf + 7, self.latent_dim))

        xavier_uniform_(self.embedding.weight.data, gain=args.gain2)
        zeros_(self.embedding.bias.data)

        for layer in self.g2z_mu:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight.data, gain=args.gain2)
                zeros_(layer.bias.data)
        for layer in self.g2z_var:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight.data, gain=args.gain2)
                zeros_(layer.bias.data)

    def forward(self, h0, label, x, edges, edge_attr, n_nodes):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["encoder_%d" % i](h, edges, x, edge_attr=edge_attr,
                                                          node_attr=h, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["encoder_%d" % i](h, edges, x, edge_attr=edge_attr,
                                                          node_attr=None, n_nodes=n_nodes)
        mu = self.g2z_mu(torch.cat((h, label), dim=1))
        log_var = self.g2z_var(torch.cat((h, label), dim=1))
        z = reparameterize(mu, log_var)
        return z

    #  没有重参数化过程
    def encode(self, h0, label, x, edges, edge_attr, n_nodes):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["encoder_%d" % i](h, edges, x, edge_attr=edge_attr,
                                                          node_attr=h, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["encoder_%d" % i](h, edges, x, edge_attr=edge_attr,
                                                          node_attr=None, n_nodes=n_nodes)
        mu = self.g2z_mu(torch.cat((h, label), dim=1))
        return mu


class Decoder(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, latent_dim, device=args.device, act_fn=nn.SiLU(), n_layers=2,
                 coords_weight=1.0, attention=False, node_attr=1, out_nf=3, padding_dim=args.qm_padding_dim):
        super(Decoder, self).__init__()
        self.padding_dim = padding_dim
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        # self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = hidden_nf
        else:
            n_node_attr = 0
        self.embedding = nn.Linear(self.latent_dim + 7, self.hidden_nf)
        self.re_nodes = MLP(nin=self.hidden_nf, nout=in_node_nf, nh=int(self.hidden_nf / 2))
        self.re_atom_position = MLP(nin=self.latent_dim + 7, nout=3, nh=int(self.padding_dim / 2))

        self.re_edge_attr = MLP(nin=self.padding_dim * (self.latent_dim + 7), nout=self.padding_dim ** 2 * 5,
                                nh=int(self.padding_dim * 2))
        self.atom_num_mlp = MLP(nin=self.padding_dim * (self.latent_dim + 7), nout=self.padding_dim, nh=self.padding_dim * 2)
        for i in range(0, n_layers):
            self.add_module("decoder_%d" % i,
                            E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                       nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=False,
                                       coords_weight=coords_weight, attention=attention, coord_renew=True))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, out_nf))

        xavier_uniform_(self.embedding.weight.data, gain=args.gain3)
        zeros_(self.embedding.bias.data)
        for layer in self.graph_dec:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight.data, gain=args.gain3)
                zeros_(layer.bias.data)
        for layer in self.re_edge_attr.net:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight.data, gain=args.gain3)
                zeros_(layer.bias.data)
        for layer in self.atom_num_mlp.net:
            if isinstance(layer, nn.Linear):
                xavier_uniform_(layer.weight.data, gain=args.gain3)
                zeros_(layer.bias.data)

    def forward(self, z, label, n_nodes, edges=None):
        z_ = torch.cat((z, label), dim=1)
        atom_num = self.atom_num_mlp(z_.view(-1, self.padding_dim * (self.latent_dim + 7)))
        h_re = self.embedding(z_.view(-1, self.latent_dim + 7)).view(-1, self.hidden_nf)
        x_re = self.re_atom_position(z_.view(-1, self.latent_dim + 7)).view(-1, 3)
        edges_re = edges
        edge_attr_re = self.re_edge_attr(z_.view(-1, n_nodes * (self.latent_dim + 7))).view(-1, 5)
        coord_re = None
        for i in range(0, self.n_layers):
            if self.node_attr:
                h_re, coord_re, edge_attr_re = self._modules["decoder_%d" % i](
                    h_re, edges_re, x_re, edge_attr=edge_attr_re, node_attr=h_re, n_nodes=n_nodes)
            else:
                h_re, coord_re, edge_attr_re = self._modules["decoder_%d" % i](
                    h_re, edges_re, x_re, edge_attr=edge_attr_re, node_attr=None, n_nodes=n_nodes)
        h_re = self.re_nodes(h_re)
        return h_re, coord_re, edge_attr_re, atom_num


class Generator(nn.Module):
    def __init__(self, nin, nout, nh):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, s):
        return F.normalize(self.net(s), p=2, dim=1)


class AREVGA(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, latent_dim, attention=True,
                 device='cpu', n_layers=2, node_attr=1, out_nf=3, padding_dim=args.qm_padding_dim):
        super(AREVGA, self).__init__()
        self.padding_dim = padding_dim
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        # Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr

        self.encoder = Encoder(in_node_nf, in_edge_nf, hidden_nf, latent_dim, device='cpu', act_fn=nn.ELU(),
                               n_layers=n_layers, coords_weight=0.3, attention=attention, node_attr=1, padding_dim=self.padding_dim)

        self.generator = Generator(
            args.latent_dim * self.padding_dim, self.latent_dim * self.padding_dim, self.latent_dim * self.padding_dim)

        self.critic = MLP(
            nin=self.padding_dim * self.latent_dim, nout=1, nh=int(self.padding_dim / 2), sigmoid=True)

        self.decoder = Decoder(in_node_nf, in_edge_nf, hidden_nf, latent_dim, device='cpu', act_fn=nn.ELU(),
                               n_layers=n_layers, coords_weight=0.3, attention=attention, node_attr=1, padding_dim=self.padding_dim)

        self.to(self.device)

    def forward(self, h0, label, x, edges, edge_attr, n_nodes):
        z = self.encoder(h0, label, x, edges, edge_attr, n_nodes=n_nodes)
        h_re, coord_re, edge_attr_re, atom_num = self.decoder(z, label, n_nodes, edges)
        return h_re, coord_re, edge_attr_re, atom_num

    def reconstruct(self, h0, label, x, edges, edge_attr, n_nodes):
        z = self.encoder.encode(h0, label, x, edges, edge_attr, n_nodes=n_nodes)
        h_re, coord_re, edge_attr_re, atom_num = self.decoder(z, label, n_nodes, edges)
        return h_re, coord_re, edge_attr_re, atom_num
