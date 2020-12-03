#%%
from typing import Tuple
import os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as tr
from PIL import Image
#%%
def load_image(image_path , to_tensor=False):
    img = Image.open(image_path)
    if to_tensor: img = tr.ToTensor()(img)
    return img

transform_list = [tr.RandomAffine(degrees=15), tr.RandomHorizontalFlip(p=0.5),  tr.RandomResizedCrop(size=(600,800), scale=(0.5,0.9))]
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
        im = load_image(os.path.join(self.data_dir, path))
        if self.transformations:
            im = self.transformations(im)
        im = tr.ToTensor()(im)
        return im, label
# %%
class CassavaDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df=None, batch_size=3, sample_size=None, data_dir = '..', num_workers=1) -> None:
        super(CassavaDataModule,self).__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.bs = batch_size
        self.sample_size = sample_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.test_df = test_df
    def train_dataloader(self, train=True, *args, **kwargs) -> DataLoader:
        weights = 1/self.train_df.groupby('label').transform('count')['image_id']
        train_dataset = CassavaDataset(self.train_df, data_dir=os.path.join(self.data_dir, 'train_images'), transformations=transformations)
        sampler = None
        if not self.sample_size: self.sample_size = len(train_dataset)
        if train: sampler = WeightedRandomSampler(weights, num_samples=self.sample_size, replacement=True)
        return DataLoader(train_dataset, batch_size=self.bs, num_workers=self.num_workers, sampler=sampler)
    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        val_dataset = CassavaDataset(self.val_df, data_dir=os.path.join(self.data_dir, 'train_images'))
        return DataLoader(val_dataset, batch_size=self.bs, num_workers=self.num_workers)
    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        test_dataset = CassavaDataset(self.test_df, data_dir=os.path.join(self.data_dir, 'test_images'))
        return DataLoader(test_dataset, batch_size=self.bs, num_workers=self.num_workers)