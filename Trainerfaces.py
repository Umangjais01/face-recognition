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

import resnet
from PIL import Image

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
import torchvision.transforms as transforms

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # Convert to single-channel grayscale
    transforms.Resize((224, 224)),                 # Resize to 224x224 pixels
    transforms.RandomRotation(degrees=15),         # Randomly rotate by 15 degrees
    transforms.RandomHorizontalFlip(p=0.5),        # Randomly flip horizontally with a 50% probability
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Randomly crop and resize to 224x224
    transforms.ToTensor(),                         # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])    # Normalize for single-channel images
])


test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # Convert to single-channel grayscale
    transforms.Resize((224, 224)),                 # Resize to 224x224 pixels
    transforms.ToTensor(),                         # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])    # Normalize for single-channel images
])
torch.manual_seed(42)

def train_model(model, train_dataloader, test_dataloader, num_epochs, device, patience=5):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    best_val_loss = float('inf')
    best_model_wts = None
    epochs_no_improve = 0

    train_time_start_resnet = timer()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_dataloader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # Validation after each epoch
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for batch in test_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += loss_fn(outputs, labels).item() * inputs.size(0)
                val_acc += accuracy_fn(outputs, labels) * inputs.size(0)

        # Calculate average losses and accuracy
        train_loss = running_loss / len(train_dataloader.dataset)
        val_loss = val_loss / len(test_dataloader.dataset)
        val_acc = val_acc / len(test_dataloader.dataset)

        # Print stats for the current epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Reduce learning rate if validation loss plateaus
        scheduler.step(val_loss)

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    train_time_end_resnet = timer()
    print(f"[INFO] Total training time: {train_time_end_resnet - train_time_start_resnet:.3f} seconds")

    # Load the best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    myfunctions1.save_model(model=model, target_dir="models", model_name="faces270524U.pt")
    return model

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
        with torch.no_grad():
            pred = resnet18(img_single.to(device))
            label_new = torch.argmax(torch.softmax(pred, dim=1), dim=1)

            print(f"Actual label: {class_names[label_single]}")
            print(f"Found label: {class_names[label_new]}")

    myfunctions1.save_model(model=resnet18, target_dir="models", model_name="faces270524U.pt")
    print("Model training and saving completed.")
    return
