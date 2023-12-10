import torch
import torch.nn as nn
import torchmetrics
import torchvision.models as models
import pytorch_lightning as pl
import torch.nn.functional as F
from ai.ResNet50Model import ResNet50Model


class finalModel(pl.LightningModule):
    
    def __init__(self, model_paths, num_classes=9, lr=3e-4):
        super(finalModel, self).__init__()
        self.num_classes = num_classes
        self.lr = lr
        
        # Load models from files
        self.models = nn.ModuleList([ResNet50Model(num_classes=num_classes), ResNet50Model(num_classes=num_classes), ResNet50Model(num_classes=num_classes)])
        features = []
        for model_path in range(3):
            #resnet_model = ResNet50Model(num_classes=num_classes)#torch.load(model_path)
            features.append(self.models[model_path].in_features2)
            # Remove the last fully connected layer (fc) from each ResNet model
            self.models[model_path].model.fc = nn.Identity()
            #self.models.append(resnet_model)
        
        # Linear layer for combined predictions
        self.fc1= nn.Linear(3 * features[0], 256)

        self.fc2 = nn.Linear(256, 128),
        self.fc3 = nn.Linear(128, self.num_classes)
             # Assuming the fc layer in ResNet has 128 output features
        
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
        out = torch.cat([out1.flatten(), out2.flatten(), out3.flatten()], dim=0)
        
        # Final linear layer for combined prediction
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
    
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