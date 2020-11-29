#%%
import pytorch_lightning as pl
import torch 
import torch.nn as nn
import torchvision as tv
import numpy as np 
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet
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
    def __init__(self, lr=1e-3, opt_freq = 300, dec_rate = 0.98, opt_upsteps = 3 ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr =lr
        self.opt_freq = opt_freq
        self.dec_rate = dec_rate
        self.opt_upsteps = opt_upsteps
        self.example_input_array = torch.ones(size = (3,3,600,800))
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc_step', self.train_accuracy(y_hat, y))
        if batch_idx % 200 == 0:
            self.logger.experiment.add_images(f'train_image', x, dataformats ='NCHW', global_step=self.global_step)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc_step', self.val_accuracy(y_hat, y))
        if batch_idx % 200 == 0:
            fig = make_classification_figure(x, y, y_hat)
            self.logger.experiment.add_figure(f'classification_val', fig, global_step=self.global_step)
        return loss
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.train_accuracy.compute())
    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log('val_acc_epoch', self.val_accuracy.compute())
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        schedule_fun = lambda step: (self.dec_rate**step)*(1+ step%self.opt_upsteps)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = [schedule_fun]),
                    'interval': 'step', 'frequency':self.opt_freq,  'monitor':'val_loss'}
        return [optimizer], [lr_scheduler]
#%%
class Resnet18(LeafClassifier):
    def __init__(self, num_classes=5, **kwargs) -> None:
        super(Resnet18, self).__init__(**kwargs)
        self.save_hyperparameters()
        self.convnet = tv.models.resnet18(pretrained=True)
        self.convnet_out = nn.Linear(1000, num_classes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.convnet(x)
        out = self.relu(out)
        out = self.convnet_out(out)
        return out

class Resnet50(LeafClassifier):
    def __init__(self, num_classes=5, **kwargs) -> None:
        super(Resnet50, self).__init__(**kwargs)
        self.save_hyperparameters()
        self.convnet = tv.models.resnet50(pretrained=True)
        self.convnet_out = nn.Linear(1000, num_classes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.convnet(x)
        out = self.relu(out)
        out = self.convnet_out(out)
        return out

class EfficientNetB1(LeafClassifier):
    def __init__(self, num_classes=5, **kwargs) -> None:
        super(EfficientNetB1, self).__init__(**kwargs)
        self.save_hyperparameters()
        self.convnet = EfficientNet.from_pretrained('efficientnet-b1')
        self.convnet_out = nn.Linear(1000, num_classes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.convnet(x)
        out = self.relu(out)
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


