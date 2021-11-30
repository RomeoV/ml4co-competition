from typing import Tuple
import torch
import torch.nn
import torch_geometric as tg
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import numpy as np

import unittest

from data_utils.dataset import MilpDataset
from data_utils.milp_data import MilpBipartiteData


class ConfigPerformanceRegressor(torch.nn.Module):
    def __init__(self, config_dim, n_gnn_layers=4, gnn_hidden_dim=64):
        super(ConfigPerformanceRegressor, self).__init__()

        self.milp_gnn = MilpGNN(
            n_gnn_layers=n_gnn_layers,
            hidden_dim=(gnn_hidden_dim, gnn_hidden_dim),
        )
        self.config_emb = ConfigEmbedding(in_dim=(4+4+4+10), out_dim=64)
        self.regression_head = RegressionHead(in_dim=5*64)

    def forward(self, instance_config_tuple, single_instance=False):
        instance_batch, config_batch = instance_config_tuple

        graph_embedding = self.milp_gnn(instance_batch)
        config_batch_as_one_hot = torch.cat(
            [F.one_hot(config_batch[:, 0].long(), num_classes=4).float(),
             F.one_hot(config_batch[:, 1].long(), num_classes=4).float(),
             F.one_hot(config_batch[:, 2].long(), num_classes=4).float(),
             F.one_hot(config_batch[:, 3].long(), num_classes=10).float()],
            axis=-1
        )  # note that one_hot is not differentiable and drops the 'requires_grad' attribute
        config_batch_as_one_hot.requires_grad_(config_batch.requires_grad)
        config_embedding = self.config_emb(config_batch_as_one_hot)

        if single_instance:
            graph_embedding = graph_embedding.repeat((config_embedding.shape[0], 1))

        x = torch.cat([graph_embedding, config_embedding], dim=-1).relu_()
        regression_pred = self.regression_head(x)
        return regression_pred


class MilpGNN(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: Tuple[int, int],
        n_gnn_layers,
    ):
        super(MilpGNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers

        self.gnns = torch.nn.ModuleList(
            [
                GNNFwd(
                    in_dim=(9, 1),
                    out_dim=hidden_dim,
                    batch_norm=True,
                )
            ]
            + [
                GNNFwd(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    batch_norm=True,
                )
                for i in range(self.n_gnn_layers - 1)
            ]
        )

        self.max_pool = tg.nn.global_max_pool

        self.attention_layer_var = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim[0], hidden_dim[0] // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim[0] // 2, 1),
        )
        self.attention_pool_var = tg.nn.GlobalAttention(self.attention_layer_var)

        self.attention_layer_cstr = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim[1], hidden_dim[1] // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim[1] // 2, 1),
        )
        self.attention_pool_cstr = tg.nn.GlobalAttention(self.attention_layer_cstr)

    def forward(self, x):
        for l in self.gnns:
            x = l(x)

        # Max pooling
        x_var_emb = self.max_pool(x.var_feats, x.var_batch_el)
        x_cstr_emb = self.max_pool(x.cstr_feats, x.cstr_batch_el)
        # Attention pooling
        x_var_emb2 = self.attention_pool_var(x.var_feats, x.var_batch_el)
        x_cstr_emb2 = self.attention_pool_cstr(x.cstr_feats, x.cstr_batch_el)
        x = torch.cat([x_var_emb, x_cstr_emb, x_var_emb2, x_cstr_emb2], axis=-1)
        return x


class GNNFwd(torch.nn.Module):
    def __init__(
        self,
        in_dim: Tuple[int, int],
        out_dim: Tuple[int, int],
        residual=False,
        batch_norm=True,
        additional_dense=False,
    ):
        super(GNNFwd, self).__init__()
        self.Conv = tg.nn.GraphConv

        self.additional_dense = additional_dense
        if additional_dense:
            self.node_encoder = torch.nn.Sequential(torch.nn.Linear(in_dim[0], in_dim[0]), torch.nn.ReLU())
            self.cstr_encoder = torch.nn.Sequential(torch.nn.Linear(in_dim[1], in_dim[1]), torch.nn.ReLU())

        # We're using 'mean' aggregation, because I've seen one single comparison and 'mean' had better performance than 'add'
        # Table 8 (Appendix) of Bengio et al: Benchmarking GNNs: https://arxiv.org/pdf/2003.00982.pdf
        self.node_gnn = self.Conv(
            in_channels=in_dim[::-1],
            out_channels=out_dim[0],
            aggr="mean",
        )
        self.cstr_gnn = self.Conv(
            in_channels=in_dim,
            out_channels=out_dim[1],
            aggr="mean",
        )
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.node_batch_norm = tg.nn.BatchNorm(in_channels=in_dim[0])
            self.cstr_batch_norm = tg.nn.BatchNorm(in_channels=in_dim[1])

        self.residual = residual
        if self.residual:
            assert in_dim == out_dim, "For residual layers, in_dim and out_dim have to match"

    def forward(self, data):
        x_node = data.var_feats
        x_cstr = data.cstr_feats

        if self.batch_norm:
            x_node = self.node_batch_norm(x_node)
            x_cstr = self.cstr_batch_norm(x_cstr)

        if self.additional_dense:
            x_node = self.node_encoder(x_node)
            x_cstr = self.cstr_encoder(x_cstr)

        edge_attr = data.edge_attr.unsqueeze(-1)

        x_node_ = self.node_gnn(x=(x_cstr, x_node), edge_index=data.edge_index, edge_weight=edge_attr)
        x_cstr_ = self.cstr_gnn(
            x=(x_node, x_cstr),
            edge_index=data.edge_index.flip(-2),
            edge_weight=edge_attr,
        )

        if self.residual:
            x_node_ = x_node_ + x_node
            x_cstr_ = x_cstr_ + x_cstr

        x_node_ = F.relu(x_node_)
        x_cstr_ = F.relu(x_cstr_)

        assert x_node.shape[-2] == x_node_.shape[-2]
        assert x_cstr.shape[-2] == x_cstr_.shape[-2]

        data_retval = MilpBipartiteData(
                var_feats = x_node_,
                cstr_feats = x_cstr_,
                edge_indices=data.edge_index,
                edge_values=data.edge_attr,
                num_nodes = data.num_nodes,
                var_batch_el = data.var_batch_el,
                cstr_batch_el = data.cstr_batch_el)

        return data_retval


class ConfigEmbedding(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=8):
        super(ConfigEmbedding, self).__init__()
        if not hidden_dim:
            hidden_dim = 64

        self.input_batch_norm = torch.nn.BatchNorm1d(in_dim)
        self.lin1 = torch.nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.lin2 = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x):
        x = self.lin1(x).relu_()
        x = self.lin2(x).relu_()
        return x


class RegressionHead(torch.nn.Module):
    def __init__(self, in_dim=2 * 8, hidden_dim=None, out_dim=2):
        super(RegressionHead, self).__init__()
        if not hidden_dim:
            hidden_dim = 4 * in_dim

        self.lin1 = torch.nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.lin2 = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x):
        x = self.lin1(x).relu_()
        x = self.lin2(x)
        mu = x[:, 0:1]
        var = x[:, 1:2].exp()  # trick to make sure std is positive
        regression_pred = torch.cat([mu, var], dim=-1)
        return regression_pred


class TestModules(unittest.TestCase):
    def test_config_embedding(self):
        ds = MilpDataset("data/output.csv", folder="train")
        dl = tg.data.DataLoader(ds, batch_size=12)
        dl_it = iter(dl)
        instance_batch, config_batch, label_batch = next(dl_it)
        config_emb = ConfigEmbedding()
        y = config_emb(config_batch)
        self.assertEqual(y.shape, (12, 8))
        self.assertTrue(y.isfinite().all())

    def test_graph_embedding(self):
        ds = MilpDataset("data/output.csv", folder="train")
        dl = tg.data.DataLoader(ds, batch_size=12)
        dl_it = iter(dl)
        instance_batch, config_batch, label_batch = next(dl_it)
        graph_emb = MilpGNN()
        y = graph_emb(instance_batch)
        self.assertEqual(y.shape, (12, 8))
        self.assertTrue(y.isfinite().all())


class TestModelEval(unittest.TestCase):
    def test_random_batch(self):
        ds = MilpDataset("data/output.csv", folder="train")
        dl = tg.data.DataLoader(ds, batch_size=8)
        dl_it = iter(dl)
        instance_batch, config_batch, label_batch = next(dl_it)
        self.assertTrue(instance_batch.var_feats.isfinite().all())
        self.assertTrue(instance_batch.cstr_feats.isfinite().all())
        self.assertTrue(config_batch.isfinite().all())

        model = ConfigPerformanceRegressor()
        pred = model((instance_batch, config_batch))

        F.mse_loss(pred[:, 0:1], label_batch)
        F.gaussian_nll_loss(pred[:, 0:1], label_batch, pred[:, 1:2])

        self.assertTrue(pred.isfinite().all(), msg=pred)
        self.assertTrue(label_batch.isfinite().all())
