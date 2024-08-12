from PIL import Image
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichModelSummary
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
import lightning as L
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
#============
#mmseg
import mmseg
from mmseg.registry import MODELS
import mmengine
from mmengine import Config
from mmseg.utils import register_all_modules
register_all_modules()

#==========
def main():
    debug=False

    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=3)
    loss_checkpoint_callback = ModelCheckpoint(
        verbose=True,
        filename=f"val_loss-" + "epoch_{epoch}-val_loss_{valid/loss:.4f}-score_{score/valid_f1:.4f}",
        monitor="valid/loss",
        mode="min",
        save_top_k=5,
        save_last=True,
        save_weights_only=True,
        auto_insert_metric_name=False,
    )
    score_checkpoint_callback = ModelCheckpoint(
        verbose=True,
        filename=f"val_score-" + "epoch_{epoch}-val_loss_{valid/loss:.4f}-socre_{score/valid_f1:.4f}",
        monitor="score/train_f1",
        save_top_k=5,
        save_weights_only=True,
        mode="max",
        auto_insert_metric_name=False,
    )

    if not debug:
        logger = WandbLogger(project="demo", name="duck1")
    else:
        logger = TensorBoardLogger("demo", name="duck1")
        
    trainer = L.Trainer(max_epochs=400, max_steps=ep*len(train_dataloader),precision="bf16-mixed", logger=logger, callbacks=[lr_monitor,loss_checkpoint_callback,score_checkpoint_callback],log_every_n_steps=10,accumulate_grad_batches=1,gradient_clip_val=1)
    
    trainer.fit(model=wrapper_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)

if __name__ == "__main__":
    seed_everything(42)
    main()
