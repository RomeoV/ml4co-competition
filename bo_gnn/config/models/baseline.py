from typing import Tuple
import torch
import torch.nn
import torch_geometric as tg
import torch.nn.functional as F
import numpy as np

import unittest

from data_utils.dataset import MilpDataset


class ConfigPerformanceRegressor(torch.nn.Module):
    def __init__(self, config_dim, n_gnn_layers=4, gnn_hidden_dim=8):
        super(ConfigPerformanceRegressor, self).__init__()

        self.milp_gnn = MilpGNN(
            n_gnn_layers=n_gnn_layers,
            hidden_dim=(gnn_hidden_dim, gnn_hidden_dim),
        )
        self.config_emb = ConfigEmbedding(in_dim=config_dim, out_dim=8)
        self.regression_head = RegressionHead(in_dim=2 * gnn_hidden_dim + 8)

    def forward(self, instance_config_tuple, single_instance=False):
        instance_batch, config_batch = instance_config_tuple
        graph_embedding = self.milp_gnn(instance_batch)
        config_embedding = self.config_emb(config_batch)

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
        self.pool = tg.nn.global_mean_pool

    def forward(self, x):
        for l in self.gnns:
            x = l(x)
        x_var = self.pool(x.var_feats, x.var_batch_el)
        x_cstr = self.pool(x.cstr_feats, x.cstr_batch_el)
        x = torch.cat([x_var, x_cstr], axis=-1)
        return x


class GNNFwd(torch.nn.Module):
    def __init__(
        self,
        in_dim: Tuple[int, int],
        out_dim: Tuple[int, int],
        residual=False,
        batch_norm=True,
    ):
        super(GNNFwd, self).__init__()
        self.Conv = tg.nn.GraphConv

        self.node_layer = self.Conv(
            in_channels=in_dim[::-1],
            out_channels=out_dim[0],
            aggr="mean",
        )
        self.cstr_layer = self.Conv(
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
            assert (
                in_dim == out_dim
            ), "For residual layers, in_dim and out_dim have to match"

    def forward(self, data):
        x_node = data.var_feats
        x_cstr = data.cstr_feats

        if self.batch_norm:
            x_node = self.node_batch_norm(x_node)
            x_cstr = self.cstr_batch_norm(x_cstr)

        edge_attr = data.edge_attr.unsqueeze(-1)

        x_node_ = self.node_layer(
            x=(x_cstr, x_node), edge_index=data.edge_index, edge_weight=edge_attr
        )
        x_cstr_ = self.cstr_layer(
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

        data.var_feats = x_node_
        data.cstr_feats = x_cstr_

        return data


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
