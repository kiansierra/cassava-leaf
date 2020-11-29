#%%
import pandas as pd 
import numpy as np 
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import  LearningRateMonitor, ModelCheckpoint, EarlyStopping
from sklearn import model_selection
from dataloaders import CassavaDataModule
from models import Resnet18, Resnet50, EfficientNetB1
#%%
# os.listdir('..')
# %%
df = pd.read_csv('../train.csv')

#%%
train_df, val_df = model_selection.train_test_split(df, test_size=0.1, random_state=42, stratify=df.label.values)
#%%
train_df.reset_index(inplace=True, drop=True)
val_df.reset_index(inplace=True, drop=True)

#%%
datamodule = CassavaDataModule(train_df, val_df, batch_size = 6)
#%%
classifier_list = [Resnet18, Resnet50, EfficientNetB1]
#%%
classifier_names = [elem.__name__.lower() for elem in classifier_list]
# %%
classifier_model_name = 'efficientnetb1'
classifier = classifier_list[classifier_names.index(classifier_model_name)]
#%%
classifier_model_dir = os.path.join('logs', classifier_model_name)
trainer_args = {'max_epochs' :8, 'profiler' : 'simple', 'precision' :16, 'gradient_clip_val' : 100, 'gpus':1 }
model_args = {'lr' : 5e-5}
load_pretrained = True
load_pretrained = os.path.exists(classifier_model_dir) and load_pretrained
checkpoints = list(filter(lambda x : '.ckpt' in x, os.listdir(classifier_model_dir))) if load_pretrained else [] 
load_pretrained = load_pretrained and len(checkpoints)>0
#%%
if load_pretrained:
     checkpoint_path = os.path.join(classifier_model_dir, checkpoints[-1])
     model = classifier.load_from_checkpoint(checkpoint_path)
else:
     model = classifier(**model_args)
print(model)
#%%
logger = TensorBoardLogger("logs", name=classifier_model_name, log_graph=True)
lr_monitor = LearningRateMonitor(logging_interval='step')
model_chkpt = ModelCheckpoint(classifier_model_dir, monitor='val_acc_epoch', save_top_k=2, verbose=True)
early_stopper = EarlyStopping(monitor='val_acc_epoch', patience=3, verbose=True)
trainer = pl.Trainer( logger=logger, callbacks = [lr_monitor, model_chkpt, early_stopper], **trainer_args)
trainer.fit(model, datamodule)     
# %%
