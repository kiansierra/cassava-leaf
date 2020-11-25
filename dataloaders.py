#%%
from typing import Tuple
import os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tr
from PIL import Image
#%%
def load_image(image_path):
    img = Image.open(image_path)
    return img

transform_list = [tr.RandomAffine(degrees=45), tr.RandomHorizontalFlip(p=0.5),  tr.RandomResizedCrop(size=(600,800), scale=(0.5,0.9))]
transformations = tr.Compose(transform_list)

def collate_fun(transformations):
    def collate(batch):
        x, y = batch
        return transformations(x), y 
    return collate
class CassavaDataset(Dataset):
    def __init__(self, df, data_dir = '../train_images', transformations=None) -> None:
        super(CassavaDataset,self).__init__()
        self.df = df
        self.data_dir = data_dir
        self.transformations = transformations
    def __len__(self) -> int:
        return len(self.df)
    def __getitem__(self, idx) -> Tuple:
        path = self.df.loc[idx, 'image_id']
        label = self.df.loc[idx, 'label']
        im = load_image(os.path.join(self.data_dir, path))
        if self.transformations:
            im = self.transformations(im)
        im = tr.ToTensor()(im)
        return im, label
# %%
class CassavaDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, batch_size=3, shuffle=True) -> None:
        super(CassavaDataModule,self).__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.bs = batch_size
        self.shuffle = shuffle
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        train_dataset = CassavaDataset(self.train_df, transformations=transformations)
        return DataLoader(train_dataset, batch_size=self.bs, shuffle = self.shuffle)
    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        val_dataset = CassavaDataset(self.val_df)
        return DataLoader(val_dataset, batch_size=self.bs)