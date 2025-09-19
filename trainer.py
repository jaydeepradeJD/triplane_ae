import torch    
from lightning.pytorch import LightningModule
import wandb
from model import MultiTriplane

class TriplaneAE(LightningModule):
    def __init__(self, args):
        super(TriplaneAE, self).__init__()
        self.model = MultiTriplane(num_objs=args.num_objs, input_dim=3, output_dim=1, triplane_resolution=args.triplane_resolution)
        
        # Only train the embeddings (triplane representations) for the single object
        if args.single_obj:
            self.model.load_state_dict(torch.load(args.decoder_ckpt)['model_state_dict'])
            self.model.embeddings.train()
            # Freeze the decoder
            self.model.net.eval()
            for param in self.model.net.parameters():
                param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr) if args.optimizer is None else args.optimizer
        self.loss_fn = torch.nn.BCEWithLogitsLoss() if args.loss_fn is None else args.loss_fn

    def forward(self, obj_idx, x):
        return self.model(obj_idx, x)
    
    def training_step(self, batch, batch_idx):
        obj_idx, x, y = batch
        logits = self(obj_idx=obj_idx, x=x)
        logits = logits.squeeze(-1)
        loss = self.loss_fn(logits, y)

        y_hat = torch.nn.functional.sigmoid(logits)
        accuracy = (y_hat.round() == y).float().mean()
        self.log('train_accuracy', accuracy, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    

    def configure_optimizers(self):
        return self.optimizer