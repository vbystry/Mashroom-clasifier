import torchvision.transforms as transforms

train_augmentations = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.3),
    transforms.RandomEqualize(0.2),
    transforms.RandomPosterize(8, p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet Mean
                         std=[0.229, 0.224, 0.225]  # ImageNet Std
                         ),
    transforms.Resize((180, 180), antialias=True),
])

val_augmentations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet Mean
                         std=[0.229, 0.224, 0.225]  # ImageNet Std
                         ),
    transforms.Resize((180, 180), antialias=True),
])
