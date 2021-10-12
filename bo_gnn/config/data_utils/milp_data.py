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
        batch_el=None,
    ):
        super(MilpBipartiteData, self).__init__()
        self.var_feats = torch.tensor(var_feats, dtype=torch.float32)
        self.cstr_feats = torch.tensor(cstr_feats, dtype=torch.float32)
        self.edge_index = torch.tensor(edge_indices.astype(np.int64), dtype=torch.long)
        self.edge_attr = torch.tensor(edge_values, dtype=torch.float32)  # .unsqueeze(1)
        # self.y = torch.tensor([label], dtype=torch.float32)
        self.num_nodes = self.var_feats.shape[1] + self.cstr_feats.shape[1]
        if batch_el is None:
            self.batch_el = torch.zeros((self.var_feats.shape[-2]), dtype=torch.long)
        else:
            self.batch_el = batch_el

    def __inc__(self, key, value):
        if key == "edge_index":
            return torch.tensor([[self.cstr_feats.size(0)], [self.var_feats.size(0)]])
        elif key == "batch_el":
            return torch.tensor([1], dtype=torch.long)
        else:
            return super().__inc__(key, value)
