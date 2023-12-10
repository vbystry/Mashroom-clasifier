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
    
class Mushrooms_for_final:
    def __init__(self, df, augmentations=None):
        self.xxs = df['xxs'].values
        self.x1xs = df['x1xs'].values
        self.xx1s = df['xx1s'].values
        self.labels = df['label'].values
        
        if augmentations is None:
            self.augmentations = transforms.Compose([transforms.Resize((180,180), antialias=True),
                                                     transforms.ToTensor()
                                                    ])
        else:
            self.augmentations = augmentations
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample_xx = self.xxs[idx]
        sample_xx = Image.open(sample_xx).convert(mode='RGB')
        sample_xx = self.augmentations(sample_xx)
        sample_x1x = self.x1xs[idx]
        sample_x1x = Image.open(sample_x1x).convert(mode='RGB')
        sample_x1x = self.augmentations(sample_x1x)
        sample_xx1 = self.xx1s[idx]
        sample_xx1 = Image.open(sample_xx1).convert(mode='RGB')
        sample_xx1 = self.augmentations(sample_xx1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return (sample_xx, sample_x1x, sample_xx1, label)

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
    return train, test, max(df['label'])+1

def get_final_dataset_from_path(dataset_path):
    ds_path = Path(dataset_path).resolve()
    paths = list(ds_path.glob('*/*'))
    classes = [path.parent.stem for path in paths]

    categories_paths =  list(ds_path.glob('*'))
    paths = [[sorted([xx for xx in list(id.glob('*'))]) for id in list(category.glob('*'))] for category in categories_paths]

    #xxs = [xx for xx in paths[:][:][0]]

    xxs = []
    x1xs = []
    xx1s = []
    for i in range(len(paths)):
        for j in range(len(paths[i])):
            try:
                xxs.append(paths[i][j][2])
                x1xs.append(paths[i][j][0])
                xx1s.append(paths[i][j][1])
            except:
                print(paths[i][j])

    classes = [path.parent.parent.stem for path in xxs]
    Counter(classes), len(classes), len(set(classes))
    df = pd.DataFrame({'xxs': xxs, 'x1xs': x1xs, 'xx1s': xx1s, 'class': classes})
    df['class'] = df['class'].astype('category')
    df['label'] = df['class'].cat.codes
    train, test = train_test_split(df, test_size=0.1, shuffle=True, random_state=1357, stratify=df['label'])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    
    return train, test, max(df['label'])+1
