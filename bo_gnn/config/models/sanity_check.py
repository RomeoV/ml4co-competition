import torch
from models.baseline import MilpGNN


class SanityCheckGNNModel(torch.nn.Module):
    def __init__(self):
        super(SanityCheckGNNModel, self).__init__()

        self.milp_gnn = MilpGNN(
            n_gnn_layers=4,
            hidden_dim=(8, 9),
        )
        self.lin = torch.nn.Linear(in_features=8 + 9, out_features=2)

    def forward(self, instance_batch):
        graph_embedding = self.milp_gnn(instance_batch)
        preds = self.lin(graph_embedding)
        mu = preds[:, 0:1]
        var = preds[:, 1:2].exp()
        return torch.cat([mu, var], axis=1)
