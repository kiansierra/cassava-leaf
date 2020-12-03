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
import argparse 
#%%
# os.listdir('..')
# %%
def train(args, trainer_args, model_args):
     df = pd.read_csv(os.path.join(args['data_directory'], 'train.csv'))
     train_df, val_df = model_selection.train_test_split(df, test_size=0.1, random_state=42, stratify=df.label.values)
     train_df.reset_index(inplace=True, drop=True)
     val_df.reset_index(inplace=True, drop=True)
     datamodule = CassavaDataModule(train_df, val_df, batch_size = args['batch_size'], data_dir=args['data_directory'], num_workers=4)
     classifier_list = [Resnet18, Resnet50, EfficientNetB1]
     classifier_names = [elem.__name__.lower() for elem in classifier_list]
     classifier_model_name = args['model_type']
     classifier = classifier_list[classifier_names.index(classifier_model_name)]
     classifier_model_dir = os.path.join('logs', classifier_model_name)
     #trainer_args = {'max_epochs' :8, 'profiler' : 'simple', 'precision' :16, 'gradient_clip_val' : 100, 'gpus':1 }
     #model_args = {'lr' : 5e-5}
     load_pretrained = True
     load_pretrained = os.path.exists(classifier_model_dir) and load_pretrained
     checkpoints = list(filter(lambda x : '.ckpt' in x, os.listdir(classifier_model_dir))) if load_pretrained else [] 
     load_pretrained = load_pretrained and len(checkpoints)>0
     if load_pretrained:
          checkpoint_path = os.path.join(classifier_model_dir, checkpoints[-1])
          model = classifier.load_from_checkpoint(checkpoint_path)
     else:
          model = classifier(**model_args)
     print(model)
     logger = TensorBoardLogger("logs", name=classifier_model_name, log_graph=True)
     lr_monitor = LearningRateMonitor(logging_interval='step')
     model_chkpt = ModelCheckpoint(classifier_model_dir, monitor='val_acc_epoch', save_top_k=2, verbose=True)
     early_stopper = EarlyStopping(monitor='val_acc_epoch', patience=6, verbose=True)
     trainer = pl.Trainer( logger=logger, callbacks = [lr_monitor, model_chkpt, early_stopper], **trainer_args)
     trainer.fit(model, datamodule)  
#%%
if __name__=="__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument('-dd', '--data_directory', type=str, default='..', help='Data directory')
     parser.add_argument('-m', '--model_type', type=str, default='efficientnetb1', help='model')
     parser.add_argument('-bs', '--batch_size', type=int, default=8, help='batch size')
     subparsers = parser.add_subparsers()
     #model_parser = subparsers.add_parser('model_arguments', add_help=False)
     model_parser = parser.add_argument_group("model_arguments")
     model_parser.add_argument('-lr', '--lr', type=float, default=5e-5, help='learning_rate')
     trainer_parser = parser.add_argument_group("trainer_arguments")
     #trainer_parser = subparsers.add_parser('trainer_arguments', add_help=False)
     trainer_parser.add_argument('-ep', '--max_epochs', type=int, default=6, help='max epochs')
     trainer_parser.add_argument('-pr', '--precision', type=int, default=16, help='precision')
     trainer_parser.add_argument('-gc', '--gradient_clip_val', type=int, default=50, help='gradient_clip_val')
     trainer_parser.add_argument('-gp', '--gpus', type=int, default=1, help='gpus')
     # trainer_args = trainer_parser.parse_args()
     # model_args = model_parser.parse_args()
     args = parser.parse_args()
     arg_groups = {}
     for group in parser._action_groups:
          group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
          arg_groups[group.title]=argparse.Namespace(**group_dict)
     train(vars(arg_groups['optional arguments']), vars(arg_groups['trainer_arguments']), vars(arg_groups['model_arguments'])  )

