# Mashroom-clasifier

## Usage

1. Create a dataset for a classifier that classificates what features are visible

```
python .\src\scripts\preprocess_features.py <original_dataset_directory_path> <output_directory_path>
```

2. Create a dataset containing images with a target feature

```
python .\src\scripts\preprocess_feature_classification.py <original_dataset_directory_path> <output_directory_path> <position_of_1>
```

3. Training the model

```
python .\src\train_cap.py <dataset_path>
```

Position of 1 signifies the position of 1 after '\_' with 1 - cap, 2 - under cap, 3 - leg

**Notice** The command above works for any dataset not just for the cap classification

## Project structure

1. **ai** folder
   - ResNet50Model.py - contains the custom LightningModule using the pretrained pytorch resnet50 model with DEFAULT weights
   - ResNet101Model.py - contains the custom LightningModule using the pretrained pytorch resnet101 model with DEFAULT weights
   - transforms.py - contains the composed transforms
   - utils.py - contains the dataset class as a function for creating a dataset from a given directory path
2. **scripts** folder
   - preprocess_features.py - a script that creates a dataset by copying images with the same features to a feature directory
   - preprocess_feature_classification.py - a script that creates a dataset by copying images with target features while maintaining the original dataset's structure
