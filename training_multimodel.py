#%%
import pandas as pd 
import numpy as np 
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import  LearningRateMonitor, ModelCheckpoint, EarlyStopping
from sklearn import model_selection
from dataloaders import CassavaDataModule
from multimodels import ClassifierEnsemberler, ResNet, WNetBase, IdentityClassifier, TransformerClassifier
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
datamodule = CassavaDataModule(train_df, val_df, batch_size = 1)
#%%

# %%
identity_model_name = 'identity'
identity_model_dir = os.path.join('logs', identity_model_name)
trainer_args = {'max_epochs' :8, 'profiler' : 'simple', 'precision' :32, 'gradient_clip_val' : 100, 'gpus':1 }
model_args = {'lr' : 5e-5}
#%%
wnet_base = WNetBase(num_channels=16)
classifier_list = [ResNet(num_blocks=1, in_channels=16, num_channels=16), ResNet(num_blocks=1, in_channels=16, num_channels=16)]
ensembler = ClassifierEnsemberler(classifier_list)
identity_model = IdentityClassifier(wnet_base, ensembler)
#%%
logger = TensorBoardLogger("logs", name=identity_model_name, log_graph=True)
lr_monitor = LearningRateMonitor(logging_interval='step')
# model_chkpt = ModelCheckpoint(identity_model_dir, monitor='val_acc_epoch', save_top_k=2, verbose=True)
# early_stopper = EarlyStopping(monitor='val_acc_epoch', patience=3, verbose=True)
callback_list = [lr_monitor]
trainer = pl.Trainer( logger=logger, callbacks = callback_list, **trainer_args, max_steps=100)
trainer.fit(identity_model, datamodule)     
#%%
classifier_model_name = 'classifier-ensemble'
classifier_model_dir = os.path.join('logs', classifier_model_name)
logger = TensorBoardLogger("logs", name=classifier_model_name, log_graph=True)
model_chkpt = ModelCheckpoint(classifier_model_dir, monitor='val_acc_epoch', save_top_k=2, verbose=True)
early_stopper = EarlyStopping(monitor='val_acc_epoch', patience=3, verbose=True)
classifier_model = TransformerClassifier(wnet_base, ensembler)
trainer = pl.Trainer( logger=logger, callbacks = [lr_monitor, model_chkpt, early_stopper], **trainer_args)
trainer.fit(identity_model, datamodule)  