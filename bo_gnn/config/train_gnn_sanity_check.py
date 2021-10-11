import torch
import tqdm
from data_utils.dataset import Problem, Folder, DataFormat, MilpDataset
from models.baseline import MilpGNN

from torch_geometric.data import Data, DataLoader, Batch


class SanityCheckGNNModel(torch.nn.Module):
    def __init__(self):
        super(SanityCheckGNNModel, self).__init__()

        self.milp_gnn = MilpGNN(
            n_gnn_layers=8,
            hidden_dim=(8, 9),
        )
        self.lin = torch.nn.Linear(in_features=8, out_features=1)

    def forward(self, instance_batch):
        graph_embedding = self.milp_gnn(instance_batch)
        head = self.lin(graph_embedding)
        return head


def main():
    problem = Problem.ONE
    model = SanityCheckGNNModel()

    data_train = DataLoader(
        MilpDataset(
            "data/max_train_data.csv",
            folder=Folder.TRAIN,
            data_format=DataFormat.MAX,
            problem=problem,
            dry=(not torch.cuda.is_available()),
            only_default_config=True,
        ),
        shuffle=True,
        batch_size=128,
        drop_last=False,
        num_workers=3,
    )
    mu, sig = (
        data_train.dataset.csv_data_full.time_limit_primal_dual_integral.mean(),
        data_train.dataset.csv_data_full.time_limit_primal_dual_integral.std(),
    )

    N_EPOCHS = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for e in tqdm.trange(N_EPOCHS):
        losses = []
        for b in data_train:
            inst, conf, lbl = b
            inst.var_feats.requires_grad_(True)
            inst.cstr_feats.requires_grad_(True)
            inst.edge_attr.requires_grad_(True)
            inst, lbl = inst.to(device), ((lbl - mu) / sig).to(device)
            pred = model(inst)

            loss = torch.nn.functional.mse_loss(pred, lbl)
            losses += [loss]
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Loss: {sum(losses)/len(losses)}")


if __name__ == "__main__":
    main()
