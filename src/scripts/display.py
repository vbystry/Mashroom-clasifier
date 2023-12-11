import os
import sys

import torch
import torchvision.transforms as transforms
from PIL import Image
from src.ai.model import finalModel

if __name__ == "__main__":
    # ----------------ZDEFINIOWANIE STAŁYCH---------------- #
    # path do checkpoint'a, z którego odczytujemy zapisany model
    checkpoint = r"example.checkpoint.please.change.accordingly.ckpt"

    # path do bazy danych, w której ponazywane są foldery z grzybami
    db_path = r"database\because\this\code\needs\labels\probably"
    # ----------------KONIEC SEKCJI DEFINIOWANIA---------------- #

    if len(sys.argv) < 3:
        raise ValueError("Required 3 paths to sources")

    # przygotowanie modelu
    finalModel = finalModel.load_from_checkpoint(checkpoint)
    finalModel.eval()

    # przygotowanie zdjęć
    photos = sys.argv[1:4]

    val_augmentations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet Mean
                             std=[0.229, 0.224, 0.225]  # ImageNet Std
                             ),
        transforms.Resize((180, 180), antialias=True),
    ])
    photos = [val_augmentations(Image.open(photo).convert(mode='RGB')).unsqueeze(0) for photo in photos]

    photo_labels = []
    for label in os.listdir(db_path):
        photo_labels.append(label)

    # podstawianie zdjęć do modelu oraz obróbka wyników
    for photo in photos:
        output = finalModel(photo, photo, photo)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_prob = probabilities[0, predicted_class].item()
        print(f'Predicted Class: {photo_labels[predicted_class]}')
        print(f'Probability: {predicted_prob:.4f}')
