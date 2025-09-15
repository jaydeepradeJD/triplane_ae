import os
import argparse
import numpy as np
import torch

from trainer import TriplaneAE
from data.data import TriPlaneDataset
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins.environments import SLURMEnvironment

def train(args):
    dataset = TriPlaneDataset(data_dir=args.data_dir, num_points=args.num_points)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    
    wandb_logger = WandbLogger(project="triplane-ae", config=args, name=args.experiment_name, save_dir=args.save_dir)
    model_checkpoint = ModelCheckpoint(dirpath=args.save_dir, filename="model_{epoch}", monitor="train_loss", mode="min", save_top_k=5, save_last=True)
    
    model = TriplaneAE(args)

    trainer = Trainer(devices=args.devices,
                      accelerator="gpu", max_epochs=args.epochs,
                      callbacks=[model_checkpoint],
                      logger=wandb_logger,
                      fast_dev_run=args.debug_run, 
                      plugins=[SLURMEnvironment(auto_requeue=False)], # use when running on slurm )
                      )
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="/work/mech-ai/jrrade/Tri-plane/aeroplane_subset")
    args.add_argument("--save_dir", type=str, default="/work/mech-ai/jrrade/Tri-plane/triplane_ae/training_logs")
    args.add_argument("--devices", type=int, default=1)
    args.add_argument("--batch_size", type=int, default=4)
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--epochs", type=int, default=100)
    args.add_argument("--num_workers", type=int, default=0)
    args.add_argument("--num_objs", type=int, default=1)
    args.add_argument("--num_points", type=int, default=500000)
    args.add_argument("--triplane_resolution", type=int, default=128)
    args.add_argument("--experiment_name", type=str, default="triplane-ae")
    args.add_argument("--debug_run", type=bool, default=False)
    args.add_argument("--optimizer", type=str, default=None)
    args.add_argument("--loss_fn", type=str, default=None)
    args = args.parse_args()

    train(args)