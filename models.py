#%%
import pytorch_lightning as pl
import torch 
import torch.nn as nn
import torchvision as tv
import numpy as np 
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
from vision_transformer_pytorch import VisionTransformer
import wandb 
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.cuda.amp import autocast
#%%
def make_classification_figure(image_tensor, y_true, y_pred):    
    batch_tensor = image_tensor.permute((0,2,3,1)).cpu().numpy()
    batch_tensor = batch_tensor/batch_tensor.max()
    batch_tensor = np.clip(batch_tensor, a_min=0.001, a_max=0.999)
    num_images = batch_tensor.shape[0]
    preds = torch.argmax(y_pred, dim=1).cpu().numpy()
    fig, axs = plt.subplots(ncols=num_images, figsize = (10,10 ))
    if num_images > 1:
        for num in range(num_images):
            axs[num].imshow(batch_tensor[num])
            axs[num].set_title(f"True: {y_true[num]} -- Pred: {preds[num]}")
    else:
        axs.imshow(batch_tensor[0])
        axs.set_title(f"True: {y_true[0]} -- Pred: {preds[0]}")
    return fig

class LeafClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3, opt_freq = 300, dec_rate = 0.98, opt_upsteps = 3, weight_decay=0 ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr =lr
        self.opt_freq = opt_freq
        self.dec_rate = dec_rate
        self.opt_upsteps = opt_upsteps
        self.wd = weight_decay 
        self.example_input_array = torch.ones(size = (1,3,384,384))
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        #self.train_fscore = pl.metrics.FBeta(num_classes=5)
    def log_step(self, phase, loss, acc, x, y, y_pred, batch_idx):
        self.log(f'{phase}_loss', loss)
        self.log(f'{phase}_acc_step', acc, prog_bar=True)
        #self.log('train_fscore', self.train_fscore(y_hat, y))
        if batch_idx % 200 == 0 and isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_images(f'{phase}_image', x, dataformats ='NCHW', global_step=self.global_step)
        elif batch_idx % 200 == 0 and isinstance(self.logger, WandbLogger):
            batch_imgs = x.data.permute(0,2,3,1).cpu().numpy()
            true_labels = y.data.cpu().numpy()
            pred_labels = torch.argmax(y_pred.data, dim=1).cpu().numpy()
            self.logger.experiment.log({f'{phase}_image':[wandb.Image(img, caption=f'True: {true_label} -- Pred: {pred_label}') for img, true_label, pred_label in zip(batch_imgs, true_labels, pred_labels)]})
            # for name, param in self.named_parameters():
            #     self.logger.experiment.log({name:[wandb.Image()]})

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.train_accuracy(y_hat, y)
        self.log_step('train', loss,acc,x, y, y_hat, batch_idx)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.val_accuracy(y_hat, y)
        self.log_step('val', loss,acc,x, y, y_hat, batch_idx)
        return loss
    def test_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        return {'label':torch.argmax(y_hat,axis=1)}
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.train_accuracy.compute())
        #self.log('train_fscore_epoch', self.train_fscore.compute())
    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log('val_acc_epoch', self.val_accuracy.compute())
    def test_epoch_end(self, outputs):
        results = torch.cat([elem['label'] for elem in outputs])
        self.test_results = results
        return results
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # schedule_fun = lambda step: (self.dec_rate**step)*(1+ step%self.opt_upsteps)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = [schedule_fun])
        lr_scheduler = {'scheduler':scheduler, 'interval': 'step', 'frequency':self.opt_freq,  'monitor':'val_loss'}
        return [optimizer], [lr_scheduler]
#%%
class Resnet18(LeafClassifier):
    def __init__(self, num_classes=5, **kwargs) -> None:
        super(Resnet18, self).__init__(**kwargs)
        self.save_hyperparameters()
        self.convnet = tv.models.resnet18(pretrained=False)
        self.dropout = nn.Dropout(inplace=True)
        self.convnet_out = nn.Linear(1000, num_classes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.convnet(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.convnet_out(out)
        return out

class Resnet50(LeafClassifier):
    def __init__(self, num_classes=5, **kwargs) -> None:
        super(Resnet50, self).__init__(**kwargs)
        self.save_hyperparameters()
        self.convnet = tv.models.resnet50(pretrained=False)
        self.dropout = nn.Dropout(inplace=True)
        self.convnet_out = nn.Linear(1000, num_classes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.convnet(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.convnet_out(out)
        return out

class EfficientNetBase(LeafClassifier):
    def __init__(self, base_name = 'efficientnet-b5', num_classes=5, **kwargs) -> None:
        super(EfficientNetBase, self).__init__(**kwargs)
        self.save_hyperparameters()
        # self.convnet = EfficientNet.from_pretrained('efficientnet-b1')
        self.convnet = EfficientNet.from_name(base_name)
        self.dropout = nn.Dropout(inplace=True)
        #self.convnet = EfficientNet(blocks_args=2)
        self.convnet_out = nn.Linear(1000, num_classes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.convnet(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.convnet_out(out)
        return out

class VisTransformer(LeafClassifier):
    def __init__(self, num_classes=5, **kwargs) -> None:
        super(VisTransformer, self).__init__(**kwargs)
        self.save_hyperparameters()
        # self.convnet = EfficientNet.from_pretrained('efficientnet-b1')
        self.convnet = VisionTransformer.from_pretrained('ViT-B_16')   
        #self.convnet = VisionTransformer(image_size=(608,800), patch_size=(19,25), num_layers=8)
        self.dropout = nn.Dropout(inplace=True)
        #self.convnet = EfficientNet(blocks_args=2)
        self.convnet_out = nn.Linear(1000, num_classes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.convnet(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.convnet_out(out)
        return out

class EnsembleClassifier(LeafClassifier):
    def __init__(self, num_classes=5, **kwargs):
        super(EnsembleClassifier, self).__init__(**kwargs)
        self.model1 = VisionTransformer.from_pretrained('ViT-B_16')  
        #self.model1.load_state_dict(torch.load('../input/vit-model-1/ViT-B_16.pt'))
        self.model2 = EfficientNet.from_pretrained('efficientnet-b3')
        self.dropout = nn.Dropout(p=0.75, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.convnet_out = nn.Linear(1000, num_classes)
    # @autocast()
    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        out = self.dropout(0.6 * x1 + 0.4 * x2)
        out = self.relu( out)
        out = self.convnet_out(out)
        return out
    

#%%
# class LeafClassifierModel(pl.LightningModule):
#     def __init__(self, num_classes=5, lr=1e-3) -> None:
#         super(LeafClassifierModel, self).__init__()
#         self.save_hyperparameters()
#         self.lr =lr
#         self.example_input_array = torch.ones(size = (3,3,600,800))
#         self.convnet = tv.models.resnet18(pretrained=True)
#         self.convnet_out = nn.Linear(1000, num_classes)
#         self.relu = nn.ReLU(inplace=True)
#         self.loss = nn.CrossEntropyLoss()
#         self.train_accuracy = pl.metrics.Accuracy()
#         self.val_accuracy = pl.metrics.Accuracy()
#     def forward(self, x):
#         out = self.convnet(x)
#         out = self.relu(out)
#         out = self.convnet_out(out)
#         return out
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.loss(y_hat, y)
#         self.log('train_loss', loss)
#         self.log('train_acc_step', self.train_accuracy(y_hat, y))
#         return loss
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self(x)
#         loss = self.loss(y_hat, y)
#         self.log('val_loss', loss)
#         self.log('val_acc_step', self.val_accuracy(y_hat, y))
#         return loss
#     def training_epoch_end(self, outs):
#         # log epoch metric
#         self.log('train_acc_epoch', self.train_accuracy.compute())
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         # optimizer_clf = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
#         schedule_fun = lambda epoch: 0.98**epoch
#         lr_scheduler = {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = [schedule_fun]),
#                     'interval': 'step', 'frequency':300,  'monitor':'val_loss'}
#         return [optimizer], [lr_scheduler]


