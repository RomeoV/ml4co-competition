from typing import Tuple
import torch
import torch.nn
import torch_geometric as tg
from torch_geometric.data import DataLoader
from data_utils.dataset import Folder, DataFormat, Problem, Mode
import torch.nn.functional as F
import numpy as np
import time

import unittest

from data_utils.dataset import MilpDataset


class ConfigPerformanceRegressor(torch.nn.Module):
    def __init__(self, config_dim, n_gnn_layers=4, gnn_hidden_dim=64):
        super(ConfigPerformanceRegressor, self).__init__()

        self.milp_gnn = MilpGNN(
            n_gnn_layers=n_gnn_layers,
            hidden_dim=(gnn_hidden_dim, gnn_hidden_dim),
        )
        self.config_emb = ConfigEmbedding(in_dim=config_dim, out_dim=8)
        self.regression_head = RegressionHead(
            in_dim=4 * gnn_hidden_dim, hidden_dim=8 * gnn_hidden_dim, out_dim=config_dim
        )

    def forward(self, instance_batch):
        graph_embedding = self.milp_gnn(instance_batch)
        mu_pred, sig_pred = self.regression_head(graph_embedding)
        return torch.stack((mu_pred, sig_pred), axis=-1)


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

        self.out_dim = out_dim

        self.lin1 = torch.nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.lin2 = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.lin3_mu = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim)
        self.lin3_sig = torch.nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x):
        x = self.lin1(x).relu_()
        x = self.lin2(x).relu_()
        mu = self.lin3_mu(x)
        sig = self.lin3_sig(x).exp()

        return mu, sig


class TestModules(unittest.TestCase):
    def test_graph_embedding(self):
        dry = True
        problem = Problem.ONE
        dl = DataLoader(
            MilpDataset(
                "data/exhaustive_dataset_all_configs/1_item_placement_results_validation.csv",
                folder=Folder.VALID,
                data_format=DataFormat.MAX,
                mode=Mode.VALID,
                problem=problem,
                dry=dry,
                instance_dir=f"{'../..' if dry else ''}/instances/{problem.value}/{Folder.VALID.value}",
            ),
            shuffle=False,
            batch_size=2,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
        )
        dl_it = iter(dl)
        instance_batch, label_batch, _instance_num = next(dl_it)
        graph_emb = MilpGNN((32, 28), 3)
        y = graph_emb(instance_batch)
        self.assertEqual(y.shape, (2, 2 * 32 + 2 * 28))
        self.assertTrue(y.isfinite().all())


class TestModelEval(unittest.TestCase):
    def test_random_batch(self):
        dry = True
        problem = Problem.ONE
        dl = DataLoader(
            MilpDataset(
                "data/exhaustive_dataset_all_configs/1_item_placement_results_validation.csv",
                folder=Folder.VALID,
                data_format=DataFormat.MAX,
                mode=Mode.VALID,
                problem=problem,
                dry=dry,
                instance_dir=f"{'../..' if dry else ''}/instances/{problem.value}/{Folder.VALID.value}",
            ),
            shuffle=False,
            batch_size=2,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
        )
        dl_it = iter(dl)
        instance_batch, label_batch, _instance_num = next(dl_it)
        self.assertTrue(instance_batch.var_feats.isfinite().all())
        self.assertTrue(instance_batch.cstr_feats.isfinite().all())

        model = ConfigPerformanceRegressor(config_dim=7).eval()
        for _ in range(3):
            pred = model(instance_batch.clone())

        self.assertEqual(pred.shape, (2, 7, 2), msg=pred)
        self.assertTrue(pred.isfinite().all(), msg=pred)
        self.assertTrue(label_batch.isfinite().all())


class TimeModelEval(unittest.TestCase):
    def test_random_batch_timing_cpu(self):
        dry = True
        problem = Problem.ONE
        dl = DataLoader(
            MilpDataset(
                "data/exhaustive_dataset_all_configs/1_item_placement_results_validation.csv",
                folder=Folder.VALID,
                data_format=DataFormat.MAX,
                mode=Mode.VALID,
                problem=problem,
                dry=dry,
                instance_dir=f"{'../..' if dry else ''}/instances/{problem.value}/{Folder.VALID.value}",
            ),
            shuffle=False,
            batch_size=2,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
        )
        dl_it = iter(dl)
        instance_batch, _label_batch, _instance_num = next(dl_it)

        for device in ["cpu", "cuda:0"]:
            model = ConfigPerformanceRegressor(config_dim=60).to(device).eval()
            instance_batch = instance_batch.to(device)

            N_warmup = 20
            N_rep = 100

            for _ in range(N_warmup):
                pred = model(instance_batch.clone())

            start_time = time.time()
            for _ in range(N_rep):
                pred = model(instance_batch.clone())
            time_diff = time.time() - start_time

            print(f"{N_rep} {device} evals took {time_diff} seconds")
