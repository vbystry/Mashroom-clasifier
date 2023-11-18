import torch
import torch.nn as nn
import torchmetrics
import torchvision.models as models
import pytorch_lightning as pl

class ResNet101Model(pl.LightningModule):
    
    def __init__(self, in_channels = 3, num_classes = 7, lr=3e-4, freeze=False):
        super(ResNet101Model, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lr = lr
        
        self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 128),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.train_acc = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes)
        self.val_acc = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes)
        
        
    def forward(self, x):
        return self.model(x)
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        return [optimizer], [scheduler]
    
    
    def training_step(self, batch, batch_idx):
        
        x, y = batch
        
        preds = self.model(x)
        
        loss = self.loss_fn(preds, y)
        self.train_acc(torch.argmax(preds, dim=1), y)
        
        self.log('train_loss', loss.item(), on_epoch=True)
        self.log('train_acc', self.train_acc, on_epoch=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        
        x,y = batch
        
        preds = self.model(x)
        
        loss = self.loss_fn(preds, y)
        self.val_acc(torch.argmax(preds, dim=1), y)
        
        self.log('val_loss', loss.item(), on_epoch=True)
        self.log('val_acc', self.val_acc, on_epoch=True)
        
    
    def test_step(self, batch, batch_idx):
        
        x,y = batch
        preds = self.model(x)
        self.test_acc(torch.argmax(preds, dim=1), y)
        
        self.log('test_acc', self.test_acc, on_epoch=True)