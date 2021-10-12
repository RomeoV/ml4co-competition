import torch
import tqdm
from data_utils.dataset import Problem, Folder, DataFormat, MilpDataset
from models.baseline import MilpGNN

from torch_geometric.data import Data, DataLoader, Batch


class SanityCheckGNNModel(torch.nn.Module):
    def __init__(self):
        super(SanityCheckGNNModel, self).__init__()

        self.milp_gnn = MilpGNN(
            n_gnn_layers=4,
            hidden_dim=(8, 9),
        )
        # self.lin_mu = torch.nn.Linear(in_features=8 + 9, out_features=1)
        # self.lin_var = torch.nn.Linear(in_features=8 + 9, out_features=1)
        self.lin = torch.nn.Linear(in_features=8 + 9, out_features=2)

    def forward(self, instance_batch):
        graph_embedding = self.milp_gnn(instance_batch)
        preds = self.lin(graph_embedding)
        mu = preds[:, 0:1]
        var = preds[:, 1:2].exp()
        return torch.cat([mu, var], axis=1)


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
        batch_size=64,
        drop_last=False,
        num_workers=3,
        pin_memory=(torch.cuda.is_available()),
    )
    mu, sig = (
        data_train.dataset.csv_data_full.time_limit_primal_dual_integral.mean(),
        data_train.dataset.csv_data_full.time_limit_primal_dual_integral.std(),
    )

    N_EPOCHS = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=5e-4)

    for e in tqdm.trange(N_EPOCHS):
        losses_nll = []
        losses_l1 = []
        losses_l2 = []
        for b in data_train:
            inst, conf, lbl = b
            inst.var_feats.requires_grad_(True)
            inst.cstr_feats.requires_grad_(True)
            inst.edge_attr.requires_grad_(True)
            inst, lbl = inst.to(device), ((lbl - mu) / sig).to(device)
            pred = model(inst)
            pred_mu = pred[:, 0:1]
            pred_var = pred[:, 1:2]

            loss_nll = torch.nn.functional.gaussian_nll_loss(pred_mu, lbl, pred_var)
            loss_l1 = torch.nn.functional.l1_loss(pred_mu, lbl)
            loss_l2 = torch.nn.functional.mse_loss(pred_mu, lbl)
            losses_nll += [loss_nll.item()]
            losses_l1 += [loss_l1.item()]
            losses_l2 += [loss_l2.item()]
            opt.zero_grad()
            loss_nll.backward()
            opt.step()
        print(
            f"   NLL-Loss: {sum(losses_nll)/len(losses_nll)} - L1-Loss: {sum(losses_l1)/len(losses_l1)} - L2-Loss: {sum(losses_l2)/len(losses_l2)}"
        )


if __name__ == "__main__":
    main()
