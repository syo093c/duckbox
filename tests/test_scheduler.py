import torch
from torch import nn
import lightning as L
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset,DataLoader
from tqdm.auto import tqdm
import ipdb
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.optim import AdamW

import sys
sys.path.append('../misc')
from learning_rates import AnnealingLR

class ToyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model=nn.Linear(8,1)
        self.loss=nn.BCEWithLogitsLoss()
        self.learning_rate=1

    def forward(self,x):
        return self.model(x)

    def training_step(self,batch):
        input=batch['data']
        target=batch['target']
        output=self.model(input)
        loss=self.loss(output,target)
        return loss

    def validation_step(self,batch):
        pass

    def configure_optimizers(self):
        train_steps = self.trainer.max_steps # note: set at trainer init
        optimizer = AdamW(self.parameters(), lr=self.learning_rate,betas=(0.9, 0.999), weight_decay=0.05)    
        lr_scheduler = AnnealingLR(
            optimizer=optimizer,
            #start_lr=self.learning_rate,
            start_lr=1,
            warmup_iter=int(train_steps * 0.03 / self.trainer.accumulate_grad_batches),
            last_iter=int(train_steps*0.6/self.trainer.accumulate_grad_batches),
            total_iters=int(train_steps/self.trainer.accumulate_grad_batches),
            min_lr=0.5,
            decay_style='cosine',
        )
        return [optimizer], [
            {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        ]

class ToyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data=[torch.rand(8) for i in range(128)]
        self.target=[torch.zeros(1) for i in range(128)]
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, index):
        data=self.data[index]
        target=self.target[index]
        return {'data':data,'target':target}


def main():
    model=ToyModel()
    dataset=ToyDataset()
    train_dl=DataLoader(dataset=dataset,batch_size=8,shuffle=True,num_workers=8)
    
    ep=100
    logger = TensorBoardLogger("demo", name="duck1")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = L.Trainer(max_epochs=ep, max_steps=ep*len(train_dl),precision="bf16-mixed", logger=logger, callbacks=[lr_monitor,],log_every_n_steps=1,accumulate_grad_batches=1,gradient_clip_val=1)  
    trainer.fit(model=model,train_dataloaders=train_dl,val_dataloaders=None)

if __name__ == '__main__':
    main()
