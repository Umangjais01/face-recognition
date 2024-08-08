import base64
from io import BytesIO
import tkinter as tk
from tkinter.filedialog import askopenfilename
import torch
import torchvision.transforms as transforms
from PIL import Image
import myfunctions1
import matplotlib.pyplot as plt
from pathlib import Path
import requests
import os
import resnet
from helper_functions import accuracy_fn
from torch.utils.data import DataLoader

# Define constants
THRESHOLD = 0.7
NUM_WORKERS = 0
BATCH_SIZE = 32
CHANNELS = 1

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define test data transformations
test_transforms = transforms.Compose([
    transforms.Resize((myfunctions1.PICSIZE, myfunctions1.PICSIZE)),
    transforms.ToTensor(),
    transforms.Grayscale()
])

# Load test data
test_dir = "facesplit/val"
test_data = myfunctions1.ImageFolderCustom(targ_dir=test_dir, transform=test_transforms)
class_names = test_data.classes

# Load the pre-trained ResNet model
loaded_resnet = resnet.ResNet18(CHANNELS, resnet.ResBlock, outputs=len(class_names))
target_dir = "models"
target_dir_path = Path(target_dir)
model_name = "faces270524U.pt"
model_load_path = "models/faces270524U.pt"
if not os.path.exists(model_load_path):
    raise FileNotFoundError(f"The file does not exist: {model_load_path}")

loaded_resnet.fc = torch.nn.Linear(512, 10)  # Assuming the saved model had 10 classes
loaded_resnet.load_state_dict(torch.load(f=model_load_path, map_location=torch.device('cpu')))
loaded_resnet = loaded_resnet.to(device)

# Define the base64 string (replace this with your actual base64 string)


def predict_from_base64(base64_string):

    # Decode base64 string into bytes
    image_data = base64.b64decode(base64_string)

    # Load image from bytes
    image = Image.open(BytesIO(image_data))

    # image = Image.open(BytesIO(image_data))

    # Display the image
    # image.show()

    print("OPENNNN IMGE")
    plt.imshow(image)
    plt.axis('off')  # Hide the axis
    plt.title("Input Image")
    plt.show()

    # Transform the image and prepare for inference
    tensor1 = test_transforms(image)
    numpy_image = tensor1.permute(1, 2, 0).numpy()
    img_single = torch.unsqueeze(tensor1, dim=0).to(device)

    # Perform a forward pass on the image
    loaded_resnet.eval()
    with torch.inference_mode():
        pred = loaded_resnet(img_single)

    # Calculate prediction probabilities and label
    soft = torch.softmax(pred, dim=1)
    label_new = torch.argmax(soft, dim=1)
    max_prob = soft[0, label_new].cpu().item()

    if max_prob < THRESHOLD:
        idname = "Unknown" + str(max_prob.numpy())
        print(idname)
        return "Unknown"
    else:
        idname =  str(max_prob)
        print("percentage ",idname)   
        # return class_names[label_new]
        return idname


    # Display the image and prediction
    # print(class_names[label_new])
    # plt.title(idname)
    # plt.axis(False)
    # plt.imshow(numpy_image)
    # plt.show()
  
