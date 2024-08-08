from typing import Dict, List, Tuple
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timeit import default_timer as timer 
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import requests
from pathlib import Path
import myfunctions1
import os
import splitfolders  # to split dataset

# Now you can use functions or classes from split_folders

import resnet
from PIL import Image
from typing import Tuple, Dict, List


device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
BATCH_SIZE = 32
CHANNELS = 1
NUM_EPOCHS = 87

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

# Import accuracy metric
from helper_functions import accuracy_fn

train_transforms = transforms.Compose([
    transforms.Resize((myfunctions1.PICSIZE, myfunctions1.PICSIZE)),
    transforms.ToTensor(),
    transforms.Grayscale()
])

test_transforms = transforms.Compose([
    transforms.Resize((myfunctions1.PICSIZE, myfunctions1.PICSIZE)),
    transforms.ToTensor(),
    transforms.Grayscale()
])

def train_model(model, train_dataloader, test_dataloader, num_epochs, device):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    train_time_start_resnet = timer()
    resnet_results = myfunctions1.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=num_epochs,
        device=device
    )
    train_time_end_resnet = timer()
    print(f"[INFO] Total training time: {train_time_end_resnet - train_time_start_resnet:.3f} seconds")

    myfunctions1.save_model(model=model, target_dir="models", model_name="faces270524U.pt")
    return resnet_results

def prepare_and_train_model():
    input_folder = "/home/umang/Desktop/working/dataset"
    output = "facesplit"
    
    # Split the dataset
    splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.8, .2))
    print("Dataset split into training and validation sets.")

    train_dir = os.path.join(output, "train")
    test_dir = os.path.join(output, "val")
    print(f"Train directory: {train_dir}, Test directory: {test_dir}")
   
  
    train_data = myfunctions1.ImageFolderCustom(targ_dir=train_dir, transform=train_transforms)
    test_data = myfunctions1.ImageFolderCustom(targ_dir=test_dir, transform=test_transforms)

    # Check if datasets are loaded correctly
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of validation samples: {len(test_data)}")

    if len(train_data) == 0:
        raise ValueError("Training dataset is empty. Please check the data path and data loading process.")
    if len(test_data) == 0:
        raise ValueError("Validation dataset is empty. Please check the data path and data loading process.")

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    class_names = train_data.classes

    torch.manual_seed(42)
    resnet18 = resnet.ResNet18(CHANNELS, resnet.ResBlock, outputs=len(class_names))
    resnet18.to(device)

    train_model(resnet18, train_dataloader, test_dataloader, NUM_EPOCHS, device)

    for i in range(1, 21):
        random_idx = torch.randint(0, len(train_data), size=[1]).item()
        img_single, label_single = train_data[random_idx]
        img_single = torch.unsqueeze(img_single, dim=0).to(device)

        resnet18.eval()
        with torch.inference_mode():
            pred = resnet18(img_single.to(device))
            label_new = torch.argmax(torch.softmax(pred, dim=1), dim=1)

            print(f"Actual label: {class_names[label_single]}")
            print(f"Found label: {class_names[label_new]}")

    myfunctions1.save_model(model=resnet18, target_dir="models", model_name="faces270524U.pt")
    print("Model training and saving completed.")
    return

