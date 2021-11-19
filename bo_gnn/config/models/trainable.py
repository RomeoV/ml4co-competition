import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from models.baseline import ConfigPerformanceRegressor
from data_utils.dataset import Problem


class MilpGNNTrainable(pl.LightningModule):
    def __init__(
        self,
        config_dim,
        optimizer,
        weight_decay,
        initial_lr,
        batch_size,
        git_hash,
        problem: Problem,
        n_gnn_layers,
        gnn_hidden_dim,
        ensemble_size,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_ensemble = torch.nn.ModuleList(
            [
                ConfigPerformanceRegressor(
                    config_dim=config_dim,
                    n_gnn_layers=n_gnn_layers,
                    gnn_hidden_dim=gnn_hidden_dim,
                )
                for i in range(ensemble_size)
            ]
        )

    def forward(self, x, single_instance=False):
        instance_batch, config_batch = x  # we have to clone those
        predictions = torch.stack(
            [
                model.forward((instance_batch.clone(), config_batch.clone()), single_instance=single_instance)
                for model in self.model_ensemble
            ],
            axis=1,
        )
        mean_mu = predictions[:, :, 0:1].mean(axis=1)
        mean_var = predictions[:, :, 1:2].mean(axis=1)
        epi_var = (predictions[:, :, 0] - mean_mu).pow(2).mean(axis=1)
        return predictions, mean_mu, mean_var, epi_var

    def training_step(self, batch, batch_idx):
        instance_batch, config_batch, label_batch = batch
        instance_batch.cstr_feats.requires_grad_(True)
        instance_batch.var_feats.requires_grad_(True)
        instance_batch.edge_attr.requires_grad_(True)
        config_batch.requires_grad_(True)

        label_batch = label_batch.unsqueeze(axis=2)  # (B, 1, 1)

        pred = self.forward((instance_batch, config_batch))[0]
        pred_mu = pred[:, :, 0:1]
        pred_var = pred[:, :, 1:2]

        nll_loss = F.gaussian_nll_loss(pred_mu, label_batch, pred_var)
        l1_loss = F.l1_loss(pred_mu, label_batch)
        l2_loss = F.mse_loss(pred_mu, label_batch)
        sigs = pred_var.mean(axis=1).sqrt()
        self.log_dict(
            {
                "train_nll_loss": nll_loss,
                "train_sigmas": sigs,
                "train_l1": l1_loss,
                "train_l2": l2_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return l2_loss

    def validation_step(self, batch, batch_idx):
        instance_batch, config_batch, label_batch = batch

        _pred, mean_mu, mean_var, epi_var = self.forward((instance_batch, config_batch))
        nll_loss = F.gaussian_nll_loss(mean_mu, label_batch, mean_var)
        l1_loss = F.l1_loss(mean_mu, label_batch)
        l2_loss = F.mse_loss(mean_mu, label_batch)
        self.log_dict(
            {
                "val_sigmas": mean_var.sqrt(),
                "val_epi_sigmas": epi_var.sqrt(),
                "val_nll_loss": nll_loss,
                "val_loss_l1": l1_loss,
                "val_loss_l2": l2_loss,
            },
            prog_bar=True,
        )

    def configure_optimizers(self):
        if self.hparams.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.initial_lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.initial_lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.initial_lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.initial_lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", verbose=True, min_lr=1e-6, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train_nll_loss",
        }
