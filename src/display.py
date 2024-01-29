import os
import sys
import numpy as np


import torch
import torchvision.transforms as transforms
from PIL import Image
from ai.model import finalModel
import torch.nn.functional as F


def check(res_array):
    for element in res_array[0]:
        if element > 0.3:
            return True
    return False

def false_positive(res_array1, res_array2, res_array3, predicted_prob):
    res_array1 = check(res_array1)
    res_array2 = check(res_array2)
    res_array3 = check(res_array3)
    
    if((res_array1 or res_array2 or res_array3) and predicted_prob < 0.9):
        return True
    else:
        return False

if __name__ == "__main__":
    # ----------------ZDEFINIOWANIE STAŁYCH---------------- #
    # path do checkpoint'a, z którego odczytujemy zapisany model
    checkpoint = r"trained_models\epoch=5-step=486.ckpt"

    # path do bazy danych, w której ponazywane są foldery z grzybami
    db_path = r"datasets\original_dataset"
    # ----------------KONIEC SEKCJI DEFINIOWANIA---------------- #

    #if len(sys.argv) < 3:
    #    raise ValueError("Required 3 paths to sources")

    # przygotowanie modelu
    model = finalModel.load_from_checkpoint(checkpoint, num_classes=23)
    model.eval()

    # przygotowanie zdjęć
    photos = sys.argv[1:4]

    val_augmentations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]
                             ),
        transforms.Resize((180, 180), antialias=True),
    ])
    photos = [val_augmentations(Image.open(photo).convert(mode='RGB')).unsqueeze(0).to(device='cuda') for photo in photos]

    photo_labels = []
    for label in os.listdir(db_path):
        photo_labels.append(label)

    # podstawianie zdjęć do modelu oraz obróbka res_arrayów
    for photo in photos:
        output, out1, out2, out3 = model(photo, photo, photo)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_prob = probabilities[0, predicted_class].item()
        print(f'Predicted Class: {photo_labels[predicted_class]}')
        print(f'Probability: {predicted_prob:.4f}')

        # Przypisanie prawdopodobieństw dla każdego modelu
        probabilities_model1 = F.softmax(out1, dim=1)
        probabilities_model2 = F.softmax(out2, dim=1)
        probabilities_model3 = F.softmax(out3, dim=1)

        probabilities_model1_np = probabilities_model1.detach().cpu().numpy()
        probabilities_model2_np = probabilities_model2.detach().cpu().numpy()
        probabilities_model3_np = probabilities_model3.detach().cpu().numpy()

        np.set_printoptions(threshold=np.inf)
    
        print("IS False Positive: ", false_positive(probabilities_model1_np, probabilities_model2_np, probabilities_model3_np, predicted_prob))
