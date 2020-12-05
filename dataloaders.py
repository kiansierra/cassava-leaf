#%%
from typing import Tuple
import os
from sklearn import model_selection
import pandas as pd 
import numpy as np 
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
from PIL import Image
#%%
from albumentations.pytorch import ToTensorV2
from albumentations import *
from torchvision import transforms as tr
#%%
cfg= {'img_size': (384,384)}

def get_train_transforms():
    return Compose([
            RandomResizedCrop(cfg['img_size'][0], cfg['img_size'][1]),
            #Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_valid_transforms():
    return Compose([
            CenterCrop(cfg['img_size'][0], cfg['img_size'][1], p=1.),
            Resize(cfg['img_size'][0], cfg['img_size'][1]),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_inference_transforms():
    return Compose([
            RandomResizedCrop(cfg['img_size'][0], cfg['img_size'][1]),
            #Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)
#%%
def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    #print(im_rgb)
    img = np.zeros(shape=(608,800,3), dtype=im_rgb.dtype)
    img[4:-4] = im_rgb
    return img        
#%%
def load_image(image_path , to_tensor=False):
    img_pil = Image.open(image_path)
    img_array = np.array(img_pil)
    img = np.zeros(shape=(608,800,3), dtype=img_array.dtype)
    img[4:-4] = img_array
    img = Image.fromarray(img)
    if to_tensor: img = tr.ToTensor()(img)
    return img

transform_list = [tr.RandomAffine(degrees=15), tr.RandomHorizontalFlip(p=0.5),  tr.RandomResizedCrop(size=(608,800), scale=(0.5,0.9))]
transformations = tr.Compose(transform_list)

def collate_fun(transformations):
    def collate(batch):
        x, y = batch
        return transformations(x), y 
    return collate
class CassavaDataset(Dataset):
    def __init__(self, df, data_dir = '..', transformations=None) -> None:
        super(CassavaDataset,self).__init__()
        self.df = df
        self.data_dir = data_dir
        self.transformations = transformations
    def __len__(self) -> int:
        return len(self.df)
    def __getitem__(self, idx) -> Tuple:
        path = self.df.loc[idx, 'image_id']
        label = self.df.loc[idx, 'label']
        im = get_img(os.path.join(self.data_dir, path))
        if self.transformations:
            im = self.transformations(image=im)['image']
        # im = tr.ToTensor()(im)
        return im, label
# %%
class CassavaDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=3, sample_size=None, data_dir = '..', num_workers=1) -> None:
        super(CassavaDataModule,self).__init__()
        self.bs = batch_size
        self.sample_size = sample_size
        self.data_dir = data_dir
        self.num_workers = num_workers
    def prepare_data(self):
        # Prepare Data
        self.df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, 'sample_submission.csv'))
        train_df, val_df = model_selection.train_test_split(self.df, test_size=0.05, random_state=42, stratify=self.df.label.values)
        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
    # def setup(self, stage):
    #     train_df, val_df = model_selection.train_test_split(self.df, test_size=0.1, random_state=42, stratify=self.df.label.values)
    #     self.train_df = train_df.reset_index(drop=True)
    #     self.val_df = val_df.reset_index(drop=True)
    def train_dataloader(self, train=True, *args, **kwargs) -> DataLoader:
        weights = 1/self.train_df.groupby('label').transform('count')['image_id']
        train_dataset = CassavaDataset(self.train_df, data_dir=os.path.join(self.data_dir, 'train_images'), transformations=get_train_transforms())
        sampler = None
        if not self.sample_size: self.sample_size = len(train_dataset)
        if train: sampler = WeightedRandomSampler(weights, num_samples=self.sample_size, replacement=True)
        return DataLoader(train_dataset, batch_size=self.bs, num_workers=self.num_workers, sampler=sampler)
    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        val_dataset = CassavaDataset(self.val_df, data_dir=os.path.join(self.data_dir, 'train_images'), transformations=get_valid_transforms())
        return DataLoader(val_dataset, batch_size=self.bs, num_workers=self.num_workers)
    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        test_dataset = CassavaDataset(self.test_df, data_dir=os.path.join(self.data_dir, 'test_images'), transformations=get_inference_transforms())
        return DataLoader(test_dataset, batch_size=self.bs, num_workers=self.num_workers)