import torch
import tqdm
from data_utils.dataset import Problem, Folder, DataFormat, MilpDataset
from models.sanity_check import SanityCheckGNNModel

from torch_geometric.data import Data, DataLoader, Batch


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
    data_valid = DataLoader(
        MilpDataset(
            "data/max_valid_data.csv",
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
        train_losses_nll = []
        train_losses_l1 = []
        train_losses_l2 = []

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
            train_losses_nll += [loss_nll.item()]
            train_losses_l1 += [loss_l1.item()]
            train_losses_l2 += [loss_l2.item()]
            opt.zero_grad()
            loss_nll.backward()
            opt.step()

        valid_losses_nll = []
        valid_losses_l1 = []
        valid_losses_l2 = []

        for b in data_valid:
            inst, conf, lbl = b
            inst.var_feats
            inst.cstr_feats
            inst.edge_attr
            inst, lbl = inst.to(device), ((lbl - mu) / sig).to(device)
            pred = model(inst)
            pred_mu = pred[:, 0:1]
            pred_var = pred[:, 1:2]

            loss_nll = torch.nn.functional.gaussian_nll_loss(pred_mu, lbl, pred_var)
            loss_l1 = torch.nn.functional.l1_loss(pred_mu, lbl)
            loss_l2 = torch.nn.functional.mse_loss(pred_mu, lbl)
            valid_losses_nll += [loss_nll.item()]
            valid_losses_l1 += [loss_l1.item()]
            valid_losses_l2 += [loss_l2.item()]
        print()
        print(
            f"   TRAIN: NLL-Loss: {sum(train_losses_nll)/len(train_losses_nll)} - L1-Loss: {sum(train_losses_l1)/len(train_losses_l1)} - L2-Loss: {sum(train_losses_l2)/len(train_losses_l2)}"
        )
        print(
            f"   VALID: NLL-Loss: {sum(valid_losses_nll)/len(valid_losses_nll)} - L1-Loss: {sum(valid_losses_l1)/len(valid_losses_l1)} - L2-Loss: {sum(valid_losses_l2)/len(valid_losses_l2)}"
        )


if __name__ == "__main__":
    main()
