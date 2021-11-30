import pickle
import torch
import numpy as np

import pytorch_lightning as pl
from torch_geometric.data import Data


class MilpBipartiteData(Data):
    def __init__(
        self,
        var_feats,
        cstr_feats,
        edge_indices,
        edge_values,
        num_nodes = None,
        var_batch_el = None,
        cstr_batch_el = None
    ):
        super(MilpBipartiteData, self).__init__()
        self.var_feats = var_feats
        self.cstr_feats = cstr_feats
        self.edge_index = edge_indices
        self.edge_attr = edge_values
        # self.y = torch.tensor([label], dtype=torch.float32)
        if num_nodes is not None:
            self.num_nodes = num_nodes
        else:
            self.num_nodes = self.var_feats.shape[1] + self.cstr_feats.shape[1]
        if var_batch_el is not None:
            self.var_batch_el = var_batch_el
        else:
            self.var_batch_el = torch.zeros((self.var_feats.shape[0]), dtype=torch.long)
        if cstr_batch_el is not None:
            self.cstr_batch_el = cstr_batch_el
        else:
            self.cstr_batch_el = torch.zeros((self.cstr_feats.shape[0]), dtype=torch.long)

    def pin_memory(self):
        self.var_feats = self.var_feats.pin_memory()
        self.cstr_feats = self.cstr_feats.pin_memory()
        self.edge_attr = self.edge_attr.pin_memory()
        self.edge_index = self.edge_index.pin_memory()
        self.var_batch_el = self.var_batch_el.pin_memory()
        self.cstr_batch_el = self.cstr_batch_el.pin_memory()

    def __inc__(self, key, value):
        if key == "edge_index":
            return torch.tensor([[self.cstr_feats.size(0)], [self.var_feats.size(0)]])
        elif key == "var_batch_el":
            return torch.tensor([1], dtype=torch.long)
        elif key == "cstr_batch_el":
            return torch.tensor([1], dtype=torch.long)
        else:
            return super().__inc__(key, value)

    @staticmethod
    def load_from_picklefile(path):
        with open(path, "rb") as infile:
            instance_description_pkl = pickle.load(infile)
            return MilpBipartiteData(
                var_feats = torch.tensor(instance_description_pkl.variable_features, dtype=torch.float32),
                cstr_feats = torch.tensor(instance_description_pkl.constraint_features, dtype=torch.float32),
                edge_indices = torch.tensor(instance_description_pkl.edge_features.indices.astype(np.int64), dtype=torch.long),
                edge_values = torch.tensor(instance_description_pkl.edge_features.values, dtype=torch.float32)  # .unsqueeze(1)
            )
