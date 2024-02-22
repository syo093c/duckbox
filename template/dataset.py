import torch
from torch.utils.data import Dataset,DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
import ipdb
import rasterio
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import albumentations as A

class DuckDataset(Dataset):
    def __init__(self,data,label,data_transforms=None):
        pass
    def __len__(self):
        return 
    def __getitem__(self,index):
        return 

class DuckTrainDataset(Dataset):

class DuckValDataset(Dataset):

class DuckTestDataset(Dataset):
