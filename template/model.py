from ema import EMAOptimizer
from torch import nn
from torch import optim
from transformers import get_cosine_schedule_with_warmup
from transformers import get_polynomial_decay_schedule_with_warmup
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

class WrapperModel(L.LightningModule):
    def __init__(self, model, learning_rate=3e-5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss=None

        # for score calculation
        self.train_pred=[]
        self.val_pred=[]
        self.train_label=[]
        self.val_label=[]


    def forward(self, input):
        output = self.model(input)
        return output

    def training_step(self, i):
        input = i["data"]
        label = i["label"]
        output = self.forward(input)
        loss = self.loss(input=output, target=label)
        self.log("train/loss", loss)

        self.train_pred.append(output.detach().cpu())
        self.train_label.append(label.detach().cpu())

        return loss

    def configure_optimizers(self):
        steps_per_ep = len(self.train_dl)
        train_steps = len(self.train_dl) * self.trainer.max_epochs  # max epouch 100
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate,betas=(0.9, 0.999), weight_decay=0.05)
        optimizer= EMAOptimizer(optimizer=optimizer,device=torch.device('cuda'))
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(steps_per_ep * self.trainer.max_epochs * 0.03/self.trainer.accumulate_grad_batches),
            num_training_steps=int(train_steps/self.trainer.accumulate_grad_batches),
        )
        return [optimizer], [
            {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        ]

    def validation_step(self, i):
        input = i["data"]
        label = i["label"]
        output = self.forward(input)

        loss = self.loss(input=output, target=label)
        self.log("valid/loss", loss)

        self.val_pred.append(output.detach().cpu())
        self.val_label.append(label.detach().cpu())
    
    def on_train_epoch_end(self):
        train_pred = (torch.cat(self.train_pred).sigmoid()>0.5).float()
        train_label = torch.cat(self.train_label)
        self.train_pred.clear()
        self.train_label.clear()

        f1_score = self.f1(train_pred,train_label)
        gc.collect()
        self.log("score/train_f1", f1_score)

    def on_validation_epoch_end(self):
        val_pred = (torch.cat(self.val_pred).sigmoid()>0.5).float()
        val_label = torch.cat(self.val_label)
        self.val_pred.clear()
        self.val_label.clear()

        f1_score = self.f1(val_pred,val_label)
        gc.collect()
        self.log("score/valid_f1", f1_score)
