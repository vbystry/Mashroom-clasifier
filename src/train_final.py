import sys
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from ai.utils import Mushrooms_for_final, get_final_dataset_from_path
from ai.ResNet101Model import ResNet101Model
from ai.transforms import train_augmentations, val_augmentations
from ai.model import finalModel

def train_model(path):
    train, test, num_classes = get_final_dataset_from_path(path)
    test_ds = Mushrooms_for_final(test)
    models_paths = '?'
    m = finalModel(model_paths=models_paths, num_classes=num_classes)
    m.model

    num_folds = 4
    kf = StratifiedKFold(num_folds)
    lr = 3e-4

    logs = dict()

    for fold, (train_fold, val_fold) in enumerate(kf.split(X=train, y=train['label']), start=1):
        train_df = train.loc[train_fold]
        val_df = train.loc[val_fold]

        train_ds = Mushrooms_for_final(train_df, augmentations=train_augmentations)
        val_ds = Mushrooms_for_final(val_df, augmentations=val_augmentations)
        test_ds = Mushrooms_for_final(test, augmentations=val_augmentations)

        print(f"\nsamples:: train: {len(train_ds)} | valid: {len(val_ds)} | test: {len(test_ds)}\n")

        train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, num_workers=8)
        val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=64, num_workers=8)
        test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=128, num_workers=8)

        model = finalModel(model_paths=models_paths, num_classes=num_classes)
        trainer = pl.Trainer(accelerator='gpu',
                             max_epochs=10,
                             callbacks=[
                                 EarlyStopping(monitor="val_loss",
                                               mode="min",
                                               patience=2,
                                               )
                             ]
                             )

        model.hparams.lr = lr

        print(f"\n\n\n{'==' * 20} FOLD {fold} / {num_folds} {'==' * 20}")

        trainer.fit(model, train_dataloader, val_dataloader)
        metrics = trainer.logged_metrics
        trainer.test(model, test_dataloader)

        logs[f'fold{fold}'] = {
            'train_loss': metrics['train_loss_epoch'].item(),
            'val_loss': metrics['val_loss'].item(),
            'train_acc': metrics['train_acc_epoch'].item(),
            'val_acc': metrics['val_acc'].item()
        }

        print(f"Train Loss: {logs[f'fold{fold}']['train_loss']} | Train Accuracy: {logs[f'fold{fold}']['train_acc']}")
        print(f"Val Loss: {logs[f'fold{fold}']['val_loss']} | Val Accuracy: {logs[f'fold{fold}']['val_acc']}")

    model.save_hyperparameters()
    fold_train_losses = [logs[f'fold{fold}']['train_loss'] for fold in range(1, num_folds + 1)]
    fold_valid_losses = [logs[f'fold{fold}']['val_loss'] for fold in range(1, num_folds + 1)]
    fold_train_accs = [logs[f'fold{fold}']['train_acc'] for fold in range(1, num_folds + 1)]
    fold_valid_accs = [logs[f'fold{fold}']['val_acc'] for fold in range(1, num_folds + 1)]
    logs_df = pd.DataFrame({
        'fold': list(range(1, num_folds + 1)),
        'train loss': fold_train_losses,
        'validation loss': fold_valid_losses,
        'train accuracy': fold_train_accs,
        'validation accuracy': fold_valid_accs
    })
    print(logs_df)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_cap.py <dataset_path>")
    else:
        train_model(sys.argv[1])