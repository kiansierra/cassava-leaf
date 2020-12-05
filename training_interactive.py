import os
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import  LearningRateMonitor, ModelCheckpoint, EarlyStopping
from dataloaders import CassavaDataModule
from models import Resnet18, Resnet50, EfficientNetB1, VisTransformer, EnsembleClassifier
# %%
args = {'data_dir':'..',  'batch_size':2, 'sample_size': None ,'num_workers':2}
datamodule = CassavaDataModule(**args)
classifier_list = [Resnet18, Resnet50, EfficientNetB1, VisTransformer, EnsembleClassifier]
classifier_names = [elem.__name__.lower() for elem in classifier_list]
classifier_model_name = 'EnsembleClassifier'.lower()
classifier = classifier_list[classifier_names.index(classifier_model_name)]
classifier_model_dir = os.path.join('logs', classifier_model_name)
trainer_args = {'max_epochs' :8, 'profiler' : 'simple', 'precision' :16, 'gradient_clip_val' : 100, 'gpus':1 }
model_args = {'lr' : 5e-5}
load_pretrained = False
load_pretrained = os.path.exists(classifier_model_dir) and load_pretrained
checkpoints = list(filter(lambda x : '.ckpt' in x, os.listdir(classifier_model_dir))) if load_pretrained else [] 
load_pretrained = load_pretrained and len(checkpoints)>0

# print(model)
# logger = TensorBoardLogger("logs", name=classifier_model_name, log_graph=True)
#%%
if __name__=="__main__":
    if load_pretrained:
        checkpoint_path = os.path.join(classifier_model_dir, checkpoints[-1])
        model = classifier.load_from_checkpoint(checkpoint_path)
    else:
        model = classifier(**model_args)
    pl.seed_everything(42)
    wandb.login(key='355d7f0e367b84fb9f8a140be052641fbd926fb5')
    logger = WandbLogger(name=classifier_model_name, save_dir='logs',offline=True)
    
    #logger = TensorBoardLogger("logs", name=classifier_model_name, log_graph=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_chkpt = ModelCheckpoint(dirpath=classifier_model_dir, monitor='val_acc_epoch', filename='{epoch}-{val_acc_epoch:.2f}', verbose=True)
    early_stopper = EarlyStopping(monitor='val_acc_epoch', patience=6, verbose=True)
    trainer = pl.Trainer( logger=logger, callbacks = [lr_monitor, model_chkpt, early_stopper], **trainer_args) 
    #%%
    # lr_finder = trainer.tuner.lr_find(model, datamodule)

    # # Results can be found in
    # lr_finder.results

    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    # #%%
    trainer.fit(model, datamodule)  
    #%%
    #datamodule.prepare_data()
    trainer.test(model, datamodule.test_dataloader())
    # %%
    datamodule.test_df['label'] = model.test_results.cpu().numpy()
    # %%
    datamodule.test_df.to_csv('submission.csv', index=False)
# %%
