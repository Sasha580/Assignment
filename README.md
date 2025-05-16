# Neural-Computing assignment: CNN Food Classifier
## Overview
This project implements a learning-based food classification system using Convolutional Neural Networks (CNNs). The aim is to categorize food images into 91 distinct classes. The model serves as the core engine of an AI-based food preference detector: by analyzing images of food a user enjoys, it identifies their culinary tastes and can be used in recommendation systems.
## Dataset
The dataset used is hosted on [Kaggle](https://www.kaggle.com/datasets/kshileeva/foodimages-nc2425) and is structured as follows:
<pre>
foodimages-nc2425/
├── train/
│   └── train/
│       ├── class_0/
│       │   ├── image_001.jpg
│       │   └── ...
│       ├── class_1/
│       └── ... (total 91 classes)
└── test/
    └── test/
        ├── class_0/
        ├── class_1/
        └── ... (total 91 classes)
</pre>

Number of training samples ~45.000, testing - ~22.000

## Data loading and augmentation
- **Training set:** 85% of the `train/train` folder is used for training. Data augmentation was applied as:  
Random horizontal flip → Random rotation (±15°) → Resize to 256×256 → Normalization with ImageNet stats  
- **Validation set:** 15% of the `train/train` folder is used for validation. Only resized and normalized (no augmentations).  
- **Test set:** resized and normalized like validation data.  
Transformations are applied using a custom `TransformedDataset` wrapper to handle different transforms on training and validation subsets.

## Model Architecture
A custom residual CNN (`FoodClassifier`) is implemented using:  
- Initial conv_block + MaxPooling
- Three BasicBlock residual layers increasing in channels: 64 → 128 → 256 → 512
- Adaptive average pooling
- Fully connected classifier: 512×4×4 → 1024 → 91

## Training config
| Hyperparameter      | Value        |
|---------------------|--------------|
| Seed                | `42`         |
| Epochs              | `70`         |
| Batch size          | `32`         |
| Input size          | `256×256`    |
| Learning rate       | `0.001`      |
| Loss                | `CrossEntropyLoss` |
| Optimizer           | `SGD` (momentum=`0.9`) |
| Weight decay        | `1e-4`       |

Training is done with (`torch.autograd.set_detect_anomaly(True)`) for better debugging of gradient flow.

## Dependencies
Install the following dependencies in your environment:  
```bash
pip install torch==2.6.0 torchvision==0.19.0 kagglehub matplotlib numpy requests
```
If running on DSLab server:  
```bash
conda create -n foodcnn python=3.9
conda activate foodcnn
pip install torch==2.6.0 torchvision==0.19.0 kagglehub numpy requests
```
## Run notebook
Unzip and enter the repository:  
`cd GROUP_24_NC2425_PA`  
Launch the notebook:  
`jupyter notebook`  
In the notebook interface, go to the **Kernel** menu and select **Restart & Run All**. This will retrain the model and regenerate logs and weights automatically.

## Output
- Trained model weights (saved under `./saved_weights/`)
- `model_[epoch].pth`: current model with highest validation accuracy
- `results.txt`: training and validation logs
- Plots of training/validation accuracy and loss over epochs

## User preference simulation
To simulate a user, 10 images from the test set were randomly selected. These were passed through trained model to predict their food categories. 
![image](https://github.com/user-attachments/assets/40e3a956-7705-4d48-abd7-a75b32f1ab6c)

The frequency of predicted categories helps build a basic taste profile. The Hugging Face Inference API with the `mistralai/Mixtral-8x7B-Instruct-v0.1` model was used to generate a natural language description of the user’s food preferences.  
Prompt was put as:  
> Given the following list of foods that a user enjoys:  
> `[food_1, food_2, ..., food_10]`  
> Write a short paragraph describing the user's food preferences.

Example LLM response:
> Based on the given list, the user appears to enjoy a variety of foods from different cuisines. They seem to have a fondness for savory dishes, as indicated by their preference for grilled salmon, chicken wings, pork chop, and chicken curry. The user also appears to enjoy international flavors, as evidenced by their choices of nachos, escargots, and clam chowder. Additionally, the user has a sweet tooth, as they have included cheesecake in their list of favorites. However, it's interesting to note that they have included pork chop twice, which suggests that it might be one of their all-time favorite dishes. Overall, the user's food preferences indicate that they have a diverse palate and enjoy both meat-based and dairy-based dishes.
## Deliverables
- Jupyter notebook
- README
- report
## Contributors
- Aleksandra Bobrova, s3660141
- Sam Chaman Zibai, s1678876
- Ryan Dorland, s3219992
- Ksenia Shileeva, s3971449
