from typing import Tuple
import torch
import torch.nn
import torch_geometric as tg
import torch.nn.functional as F
import numpy as np

import unittest

from data_utils.dataset import MilpDataset


class ConfigPerformanceRegressor(torch.nn.Module):
    def __init__(self, config_dim, n_gnn_layers=1, gnn_hidden_layers=8):
        super(ConfigPerformanceRegressor, self).__init__()

        self.milp_gnn = MilpGNN(
            n_gnn_layers=n_gnn_layers, hidden_dim=(gnn_hidden_layers, gnn_hidden_layers)
        )
        self.config_emb = ConfigEmbedding(in_dim=config_dim)
        self.regression_head = RegressionHead()

    def forward(self, instance_config_tuple, single_instance=False):
        instance_batch, config_batch = instance_config_tuple
        graph_embedding = self.milp_gnn(instance_batch)
        config_embedding = self.config_emb(config_batch)

        if single_instance:
            graph_embedding = graph_embedding.repeat((config_embedding.shape[0], 1))

        x = torch.cat([graph_embedding, 0 * config_embedding], dim=-1)
        regression_pred = self.regression_head(x)
        mu = regression_pred[:, 0:1]
        logvar = regression_pred[:, 1:2]  # trick to make sure std is positive
        regression_pred = torch.cat([mu, torch.exp(logvar)], dim=-1)
        return regression_pred


class MilpGNN(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: Tuple[int, int] = (8, 8),
        out_dim=8,
        n_gnn_layers=1,
        use_batch_norm: bool = False,
    ):
        super(MilpGNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers

        self.input_embedding = InputEmbedding(in_dim=(9, 1), out_dim=hidden_dim)
        self.gnns = torch.nn.ModuleList(
            [
                GNNFwd(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    residual=False,
                    batch_norm=use_batch_norm
                    and (i < self.n_gnn_layers - 1),  # Not for last layer
                )
                for i in range(self.n_gnn_layers)
            ]
        )
        self.pool = tg.nn.global_mean_pool
        self.out_layer = torch.nn.Linear(
            in_features=hidden_dim[0], out_features=out_dim
        )  # we use this to make sure we achieve the correct out_dim

        self.loss = F.mse_loss

    def forward(self, x):
        x = self.input_embedding(x)
        for l in self.gnns:
            x = l(x)
        x = self.pool(x.var_feats, x.batch_el)
        x = self.out_layer(x).relu_()
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
        )  # concat=True fails. We average the heads instead (also smaller model).
        self.cstr_layer = self.Conv(
            in_channels=in_dim,
            out_channels=out_dim[1],
        )
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.node_batch_norm = tg.nn.BatchNorm(in_channels=out_dim[0])
            self.cstr_batch_norm = tg.nn.BatchNorm(in_channels=out_dim[1])

        self.residual = residual
        if self.residual:
            assert (
                in_dim == out_dim
            ), "For residual layers, in_dim and out_dim have to match"

    def forward(self, data):
        x_node = data.var_feats
        x_cstr = data.cstr_feats
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

        if self.batch_norm:
            x_node_ = self.node_batch_norm(x_node_)
            x_cstr_ = self.cstr_batch_norm(x_cstr_)

        x_node_ = F.relu(x_node_)
        x_cstr_ = F.relu(x_cstr_)

        assert x_node.shape[-2] == x_node_.shape[-2]
        assert x_cstr.shape[-2] == x_cstr_.shape[-2]

        data.var_feats = x_node_
        data.cstr_feats = x_cstr_

        return data


class InputEmbedding(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(InputEmbedding, self).__init__()
        self.var_emb = torch.nn.Linear(in_features=in_dim[0], out_features=out_dim[0])
        self.cstr_emb = torch.nn.Linear(in_features=in_dim[1], out_features=out_dim[1])

    def forward(self, x):
        var_feats_ = self.var_emb(x.var_feats).relu_()
        cstr_feats_ = self.cstr_emb(x.cstr_feats).relu_()
        x.var_feats = var_feats_
        x.cstr_feats = cstr_feats_
        return x


class ConfigEmbedding(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=8, out_dim=8):
        super(ConfigEmbedding, self).__init__()

        self.input_batch_norm = torch.nn.BatchNorm1d(in_dim)
        self.lin1 = torch.nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.lin2 = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x):
        # x = self.input_batch_norm(x)
        x = self.lin1(x).relu_()
        x = self.lin2(x).relu_()
        return x


class RegressionHead(torch.nn.Module):
    def __init__(self, in_dim=2 * 8, hidden_dim=8, out_dim=2):
        super(RegressionHead, self).__init__()

        self.lin1 = torch.nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.lin2 = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x):
        x = self.lin1(x).relu_()
        x = self.lin2(x)
        return x


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
