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
    ):
        super(MilpBipartiteData, self).__init__()
        self.var_feats = torch.tensor(var_feats, dtype=torch.float32)
        self.cstr_feats = torch.tensor(cstr_feats, dtype=torch.float32)
        self.edge_index = torch.tensor(edge_indices.astype(np.int64), dtype=torch.long)
        self.edge_attr = torch.tensor(edge_values, dtype=torch.float32)  # .unsqueeze(1)
        # self.y = torch.tensor([label], dtype=torch.float32)
        self.num_nodes = self.var_feats.shape[1] + self.cstr_feats.shape[1]
        self.var_batch_el = torch.zeros((self.var_feats.shape[0]), dtype=torch.long)
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
