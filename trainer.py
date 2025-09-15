import torch    
from lightning.pytorch import LightningModule
import wandb
from model import MultiTriplane

class TriplaneAE(LightningModule):
    def __init__(self, args):
        super(TriplaneAE, self).__init__()
        self.model = MultiTriplane(args, num_objs=args.num_objs, input_dim=3, output_dim=1, triplane_resolution=args.triplane_resolution)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr) if args.optimizer is None else args.optimizer
        self.loss_fn = torch.nn.BCEWithLogitsLoss() if args.loss_fn is None else args.loss_fn

    def forward(self, object_idx, x):
        return self.model(object_idx, x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(object_idx=0, x=x)
        y_hat = y_hat.squeeze(-1)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    

    def configure_optimizers(self):
        return self.optimizer