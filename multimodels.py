#%%
from model_parts import * 
import pytorch_lightning as pl
#%%
class WNetBase(nn.Module):
    def __init__(self,num_blocks =2, num_channels =64 , in_channels=3) -> None:
        super(WNetBase, self).__init__()
        assert num_blocks > 0 
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.init_conv = DoubleConv(in_channels=in_channels, out_channels=num_channels, mid_channels= num_channels//2)
        for num in range(self.num_blocks):
            setattr(self, f"W-block_{num}", UpDownResBlock(num_channels))
    def forward(self, x):
        outputs = []
        out = self.init_conv(x)
        for num in range(self.num_blocks):
            out = getattr(self, f"W-block_{num}")(out)
            outputs.append(out)
        return outputs
#%%
class ResNet(nn.Module):
    def __init__(self,num_blocks =3,in_channels=64, num_channels =64 , num_outputs=1000) -> None:
        super(ResNet, self).__init__()
        assert num_blocks > 0 
        self.num_blocks = num_blocks
        self.init_conv = DoubleConv(in_channels=in_channels, out_channels=num_channels, mid_channels= num_channels//2)
        for num in range(self.num_blocks):
            setattr(self, f"resblock-1_{num}", ResidualBlock(num_channels))
            setattr(self, f"resblock-2_{num}", ResidualBlock(num_channels))
            setattr(self, f"pooling_{num}", MaxAvgPool2d(kernel_size=2))
            setattr(self, f"dc_{num}", DoubleConv(in_channels=num_channels, out_channels=num_channels*2))
            num_channels*=2
        self.adapt_pool_1 = MaxAvgPoolAdaptative2d(output_size=(7,7))
        self.adapt_pool_2 = MaxAvgPoolAdaptative2d(output_size=(1,1)) 
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=num_channels, out_features=num_outputs)
    def forward(self, x):
        out = self.init_conv(x)
        for num in range(self.num_blocks):
            out = getattr(self, f"resblock-1_{num}")(out)
            out = getattr(self, f"resblock-2_{num}")(out)
            out = getattr(self, f"pooling_{num}")(out)
            out = getattr(self, f"dc_{num}")(out)
        out = self.adapt_pool_1(out)
        out = self.adapt_pool_2(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out
#%%
class ClassifierEnsemberler(nn.Module):
    def __init__(self, classifiers, num_classes=5, classifier_outputs=1000) -> None:
        super(ClassifierEnsemberler, self).__init__()
        self.num_classifiers = len(classifiers)
        for num, elem in enumerate(classifiers):
            setattr(self, f"classifier_{num}", elem) 
        self.linear = nn.Linear(in_features=self.num_classifiers*classifier_outputs, out_features=num_classes)
    def forward(self, x):
        classifier_outputs = []
        for num, elem in enumerate(x):
            out = getattr(self, f"classifier_{num}")(elem)
            classifier_outputs.append(out)
        out = torch.cat(classifier_outputs, dim=1)
        out = self.linear(out)
        return out
#%%
class IdentityClassifier(pl.LightningModule):
    def __init__(self, wnet_base, classifier_ensemble, id_weight=0.9, lr=1e-3, opt_freq = 300, dec_rate = 0.98, opt_upsteps = 3) -> None:
        assert wnet_base.num_blocks == classifier_ensemble.num_classifiers
        super(IdentityClassifier, self).__init__()
        self.lr =lr
        self.opt_freq = opt_freq
        self.dec_rate = dec_rate
        self.opt_upsteps = opt_upsteps
        self.id_weight = id_weight
        self.class_weight = 1 - id_weight
        self.wnet_base = wnet_base
        self.classifier_ensemble = classifier_ensemble
        self.wnet_top = DoubleConv(in_channels=wnet_base.num_channels, out_channels=wnet_base.in_channels, mid_channels=wnet_base.num_channels)
        self.id_loss = nn.MSELoss()
        self.class_loss = nn.CrossEntropyLoss()
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        wnet_outputs = self.wnet_base(x)
        id_output = self.wnet_top(wnet_outputs[-1])
        classifier_outputs = self.classifier_ensemble(wnet_outputs)
        return id_output, classifier_outputs
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        id_output, class_output = self(x)
        id_loss = self.id_loss(x, id_output)
        class_loss = self.class_loss(class_output, y)
        total_loss = self.id_weight * id_loss + self.class_weight * class_loss
        self.log('train_id_loss', id_loss)
        self.log('train_class_loss', class_loss)
        self.log('train_loss', total_loss)
        self.log('train_acc_step', self.train_accuracy(class_output, y))
        if batch_idx % 200 == 0 and optimizer_idx ==0:
            self.logger.experiment.add_images('train_images', x)
            self.logger.experiment.add_images('train_identity', id_output)
        return total_loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        id_output, class_output = self(x)
        id_loss = self.id_loss(x, id_output)
        class_loss = self.class_loss(class_output, y)
        total_loss = self.id_weight * id_loss + self.class_weight * class_loss
        self.log('val_id_loss', id_loss)
        self.log('val_class_loss', class_loss)
        self.log('val_loss', total_loss)
        self.log('val_acc_step', self.train_accuracy(class_output, y))
        return total_loss
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.train_accuracy.compute())
    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log('val_acc_epoch', self.val_accuracy.compute())
    def configure_optimizers(self):
        optimizer_cl = torch.optim.Adam(self.classifier_ensemble.parameters(), lr=self.lr)
        optimizer_wnet = torch.optim.Adam(list(self.wnet_base.parameters()) + list(self.wnet_top.parameters() ), lr=self.lr)
        schedule_fun = lambda step: (self.dec_rate**step)*(1+ step%self.opt_upsteps)
        lr_scheduler_cl = {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer_cl, lr_lambda = [schedule_fun]),
                    'interval': 'step', 'frequency':self.opt_freq,  'monitor':'val_loss'}
        lr_scheduler_wnet = {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer_cl, lr_lambda = [schedule_fun]),
                    'interval': 'step', 'frequency':self.opt_freq,  'monitor':'val_loss'}
        return [optimizer_cl, optimizer_wnet], [lr_scheduler_cl, lr_scheduler_wnet]
#%%
class TransformerClassifier(pl.LightningModule):
    
    def __init__(self, wnet_base, classifier_ensemble, lr=1e-3, opt_freq = 300, dec_rate = 0.98, opt_upsteps = 3) -> None:
        assert wnet_base.num_blocks == classifier_ensemble.num_classifiers
        super(TransformerClassifier, self).__init__()
        self.lr =lr
        self.opt_freq = opt_freq
        self.dec_rate = dec_rate
        self.opt_upsteps = opt_upsteps
        self.wnet_base = wnet_base
        self.classifier_ensemble = classifier_ensemble
        self.id_loss = nn.MSELoss()
        self.class_loss = nn.CrossEntropyLoss()
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        wnet_outputs = self.wnet_base(x)
        out = self.classifier_ensemble(wnet_outputs)
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        class_output = self(x)
        class_loss = self.class_loss(class_output, y)
        self.log('train_class_loss', class_loss)
        self.log('train_acc_step', self.train_accuracy(class_output, y))
        return class_loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        class_output = self(x)
        class_loss = self.class_loss(class_output, y)
        self.log('val_class_loss', class_loss)
        self.log('val_acc_step', self.train_accuracy(class_output, y))
        return class_loss
    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.train_accuracy.compute())
    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log('val_acc_epoch', self.val_accuracy.compute())
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.classifier_ensemble.parameters()) + list(self.wnet_base.parameters()), lr=self.lr)
        schedule_fun = lambda step: (self.dec_rate**step)*(1+ step%self.opt_upsteps)
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = [schedule_fun]),
                    'interval': 'step', 'frequency':self.opt_freq,  'monitor':'val_loss'}
        return [optimizer], [lr_scheduler]

#%%

# test_tensor = torch.ones(size=(2, 64, 600,800))
# model = ResNet()
# out = model(test_tensor)
# print(model)
# print(out.shape)
# #%%
# wnet = WNetBase()
# test_tensor = torch.ones(size=(2, 3, 600,800))
# out = wnet(test_tensor)
# print(wnet)
# for elem in out:
#     print(elem.shape)

#%%
wnet_base = WNetBase(num_channels=32)
classifier_list = [ResNet(num_blocks=2, in_channels=32, num_channels=32), ResNet(num_blocks=1, in_channels=32, num_channels=32)]
ensembler = ClassifierEnsemberler(classifier_list)
# %%
identity_model = IdentityClassifier(wnet_base, ensembler)
# %%
