#%%
import pandas as pd 
import numpy as np 
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import  LearningRateMonitor, ModelCheckpoint
from sklearn import model_selection
from dataloaders import CassavaDataModule
from models import Resnet18, Resnet50, EfficientNetB1
#%%
os.listdir('..')
# %%
df = pd.read_csv('../train.csv')
# %%
df.head()
#%%
train_df, val_df = model_selection.train_test_split(df, test_size=0.1, random_state=42, stratify=df.label.values)
#%%
train_df.reset_index(inplace=True, drop=True)
val_df.reset_index(inplace=True, drop=True)

#%%
datamodule = CassavaDataModule(train_df, val_df, batch_size = 3)
# %%
classifier_model_name = 'resnet50'
classifier_model_dir = os.path.join('logs', classifier_model_name)
trainer_args = {'max_epochs' :8, 'profiler' : 'simple', 'precision' :32, 'gradient_clip_val' : 100, 'gpus':1 }
load_pretrained = True
#%%
if load_pretrained:
     checkpoint_dir = os.path.join('logs', classifier_model_name)
     checkpoints = [elem for elem in os.listdir(checkpoint_dir) if elem.split('.')[-1] =='ckpt']
     checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
     model = EfficientNetB1.load_from_checkpoint(checkpoint_path)
else:
     model = EfficientNetB1(lr=5e-4)
print(model)
#%%
logger = TensorBoardLogger("logs", name=classifier_model_name, log_graph=True)
lr_monitor = LearningRateMonitor(logging_interval='step')
model_chkpt = ModelCheckpoint(classifier_model_dir, monitor='val_acc_epoch', verbose=True)
trainer = pl.Trainer( logger=logger, callbacks = [ lr_monitor, model_chkpt], **trainer_args)
trainer.fit(model, datamodule)     