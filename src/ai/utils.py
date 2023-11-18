from PIL import Image
import torch
import torchvision.transforms as transforms
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split


class Mushrooms:
    def __init__(self, df, augmentations=None):
        self.paths = df['path'].values
        self.labels = df['label'].values
        
        if augmentations is None:
            self.augmentations = transforms.Compose([transforms.Resize((180,180), antialias=True),
                                                     transforms.ToTensor()
                                                    ])
        else:
            self.augmentations = augmentations
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        sample = self.paths[idx]
        sample = Image.open(sample).convert(mode='RGB')
        sample = self.augmentations(sample)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return (sample, label)
    

def get_dataset_from_path(dataset_path):
    ds_path = Path(dataset_path).resolve()
    paths = list(ds_path.glob('*/*'))
    classes = [path.parent.stem for path in paths]

    Counter(classes), len(classes), len(set(classes))
    df = pd.DataFrame({'path': paths, 'class': classes})
    df['class'] = df['class'].astype('category')
    df['label'] = df['class'].cat.codes

    train, test = train_test_split(df, test_size=0.1, shuffle=True, random_state=1357, stratify=df['label'])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    return train, test, len(classes)


