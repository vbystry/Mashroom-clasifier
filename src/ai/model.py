import torch
import torch.nn as nn
import torchmetrics
import torchvision.models as models
import pytorch_lightning as pl
import torch.nn.functional as F
from ai.ResNet50Model import ResNet50Model


class finalModel(pl.LightningModule):
    
    def __init__(self, num_classes=23, lr=3e-4):
        super(finalModel, self).__init__()
        self.num_classes = num_classes
        self.lr = lr
        
                # Load models from files
        trained_paths = [r"trained_models\epoch=5-step=330.ckpt",
                         r"trained_models\epoch=5-step=192.ckpt",
                         r"trained_models\epoch=5-step=54.ckpt"]
        helpers = []
        m = ResNet50Model(num_classes=11)
        self.features = []
        for item in trained_paths:
            helpers.append(ResNet50Model.load_from_checkpoint(item, num_classes=23))
            self.features.append(m.in_features2)
        for h in helpers:
            h.freeze()
        self.models = nn.ModuleList(helpers)

        for i in range(len(self.models)):
            self.models[i].model.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(3 * self.features[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        
        self.train_acc = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes)
        self.val_acc = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes)
        
        
    def forward(self, x1, x2, x3):
        # Forward pass through each model
        out1 = self.models[0](x1)
        out2 = self.models[1](x2)
        out3 = self.models[2](x3)
        # Flatten and concatenate the outputs
        out = torch.cat([out1.view(-1, self.features[0]), out2.view(-1, self.features[0]), out3.view(-1, self.features[0])], dim=-1)
        out = self.fc(out)
        return out, out1, out2, out3
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        return [optimizer], [scheduler]
    
    
    def training_step(self, batch, batch_idx):
        x1, x2, x3, y = batch
        preds = self(x1, x2, x3)
        
        loss = self.loss_fn(preds, y)
        self.train_acc(torch.argmax(preds, dim=1), y)
        
        self.log('train_loss', loss.item(), on_epoch=True)
        self.log('train_acc', self.train_acc, on_epoch=True)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x1, x2, x3, y = batch
        preds = self(x1, x2, x3)
        
        loss = self.loss_fn(preds, y)
        self.val_acc(torch.argmax(preds, dim=1), y)
        
        self.log('val_loss', loss.item(), on_epoch=True)
        self.log('val_acc', self.val_acc, on_epoch=True)
        
    
    def test_step(self, batch, batch_idx):
        x1, x2, x3, y = batch
        preds = self(x1, x2, x3)
        self.test_acc(torch.argmax(preds, dim=1), y)
        
        self.log('test_acc', self.test_acc, on_epoch=True)