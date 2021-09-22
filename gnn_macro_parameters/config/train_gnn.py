import pandas as pd
from comet_ml import Experiment
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import numpy as np
from typing import Optional
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import pickle
from data_utils.milp_data import MilpBipartiteData
from torch_geometric.data import Batch

from pytorch_lightning import Trainer
from models.gnn import ConfigPerformanceRegressor
from data_utils.dataset import MilpDataset
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import Callback

class MilpGNNTrainable(pl.LightningModule):
    def __init__(self, initial_lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.initial_lr = initial_lr

        self.model = ConfigPerformanceRegressor()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (instance_batch, parameters_batch, label_batch) = batch
        pred = self.forward((instance_batch, parameters_batch))
        #loss = F.gaussian_nll_loss(pred[:, 0:1], label_batch, pred[:, 1:2])
        mse = F.mse_loss(pred, label_batch)
        self.log("rmse", torch.sqrt(mse), prog_bar=True)
        return mse

    def validation_step(self, batch, batch_idx):
        (instance_batch, parameters_batch, label_batch) = batch
        pred = self.forward((instance_batch, parameters_batch))
        l1_loss = F.l1_loss(pred, label_batch)
        val_mse = F.mse_loss(pred[:, 0:1], label_batch)
        self.log_dict(
            {"val_loss": val_mse, "val_rmse": torch.sqrt(val_mse), "val_l1": l1_loss,},
            prog_bar=False,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.initial_lr)

def get_validation_statistics(validation_dataset: pd.DataFrame):
    assert validation_dataset.shape[0]%64== 0
    average_default_primal_dual_integral = []
    average_worst_primal_dual_integral = []
    average_best_primal_dual_integral = []
    default_parameters_as_one_hot = '[0 1 0 0 0 1 0 0 0 1 0 0]'
    for i in range(int(validation_dataset.shape[0] / 64)):
        primal_dual_integrals_in_range = validation_dataset.iloc[i*64:(i+1)*64]["primal_dual_integral"]
        parameters_in_range = validation_dataset.iloc[i * 64:(i + 1) * 64]["parameters"]
        average_worst_primal_dual_integral.append(primal_dual_integrals_in_range.max())
        average_best_primal_dual_integral.append(primal_dual_integrals_in_range.min())
        default_primal_dual_integral = float(validation_dataset.iloc[i * 64:(i + 1) * 64][parameters_in_range == default_parameters_as_one_hot]["primal_dual_integral"])
        average_default_primal_dual_integral.append(default_primal_dual_integral)
    print("Mean Worst Primal Dual Integral: {}".format(np.mean(average_worst_primal_dual_integral)))
    print("Mean Default Primal Dual Integral: {}".format(np.mean(average_default_primal_dual_integral)))
    print("Mean Best Primal Dual Integral: {}".format(np.mean(average_best_primal_dual_integral)))

class EvaluatePredictedParametersCallback(Callback):
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        instance_path = trainer.val_dataloaders[0].dataset.instance_path
        val_dataset_info = trainer.val_dataloaders[0].dataset.dataset_info
        best_primal_dual_integrals = []
        for i in range(val_dataset_info.shape[0]):
            if i % 64 == 0:
                if i > 0:
                    best_primal_dual_integrals.append(best_primal_dual_integral)
                best_primal_dual_integral = np.infty
                best_prediction = np.infty

            instance_file, parameters, primal_dual_integral = val_dataset_info.loc[
                i, ["instance_file", "parameters", "primal_dual_integral"]
            ]
            parameter_as_list = []
            for param_string in parameters:
                if param_string.isdigit():
                    parameter_as_list.append(int(param_string))

            parameters_tensor = torch.tensor([parameter_as_list], dtype=torch.float32)
            with open(
                    os.path.join(
                        instance_path, instance_file.replace(".mps.gz", ".pkl")
                    ),
                    "rb",
            ) as infile:
                instance_description_pkl = pickle.load(infile)
            instance_data_tensor = MilpBipartiteData(
                var_feats=instance_description_pkl.variable_features,
                cstr_feats=instance_description_pkl.constraint_features,
                edge_indices=instance_description_pkl.edge_features.indices,
                edge_values=instance_description_pkl.edge_features.values,
            )
            instance_data_tensor_as_batch = Batch.from_data_list([instance_data_tensor])
            pl_module.model.eval()
            prediction = pl_module.model.forward((instance_data_tensor_as_batch, parameters_tensor))

            if float(prediction) < best_prediction:
                best_prediction = float(prediction)
                best_primal_dual_integral = primal_dual_integral
        self.log("val_mean_primal_dual_integral", np.mean(best_primal_dual_integrals), prog_bar=True)
        print("Mean Validation Primal Dual Integral: {}".format(np.mean(best_primal_dual_integrals)))
        pl_module.model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_dataset_info_path', type=str, required=True)
    parser.add_argument('-v', '--valid_dataset_info_path', type=str, required=True)
    arguments = parser.parse_args()
    # comet_logger = CometLogger(
    #     api_key="j9BTvzjyfbD65RalgawwvOOMv",
    #     workspace="jerry-crea",  # Optional
    #     project_name="ml4co"  # Optional
    # )
    logger = TensorBoardLogger("tb_logs", "GNN")

    trainer = Trainer(max_epochs=10, gpus=1 if torch.cuda.is_available() else 0, logger=logger,
                      callbacks=[EvaluatePredictedParametersCallback()])
    model = MilpGNNTrainable(initial_lr=0.01)
    data_train = DataLoader(
        MilpDataset(arguments.train_dataset_info_path,instances_dir="/Users/jeremyscheurer/Code/ml4comp/romeo/ml4co-competition/instances/1_item_placement/train"),
        batch_size=4,
    )
    validation_dataset = MilpDataset(arguments.valid_dataset_info_path, instances_dir="/Users/jeremyscheurer/Code/ml4comp/romeo/ml4co-competition/instances/1_item_placement/train")
    get_validation_statistics(validation_dataset.dataset_info)
    data_valid = DataLoader(
        validation_dataset,
        batch_size=4,
        num_workers=12
    )
    trainer.fit(model, train_dataloaders=data_valid, val_dataloaders=data_valid, )


if __name__ == "__main__":
    main()
