from torch import nn
from U_Model.U_util import *
from U_Model.U_Layer import E_GCL_mask


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0,
                 attention=False, node_attr=1, out_nf=3):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        # Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i,
                            E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                       nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True,
                                       coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, out_nf))
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, __ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                       node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, __ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                       node_attr=None, n_nodes=n_nodes)

        h = self.node_dec(h)  # batch_size * padding_dim, hidden_nf
        h = h * node_mask  # batch_size * padding_dim, hidden_nf
        h = h.view(-1, n_nodes, self.hidden_nf)  # batch_size, padding_dim, hidden_nf
        h = torch.sum(h, dim=1)  # batch_size, hidden_nf
        pred = self.graph_dec(h)
        return pred.squeeze(1)
