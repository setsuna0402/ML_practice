import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
import torchvision.models as models
from torchinfo import summary

# This is for the progress bar.
from tqdm.auto import tqdm
import time
from Model_Class import Classifier_Resnet18, Classifier_Resnet34

allow_device = True  # Set to False if you want to force using CPU.
use_pin_memory = True  # Set to True if you use GPU. False for CPU and MPS.
batch_size = 128
# 0 means only the main process will load data. Greater than 0 means number of subprocesses to use for data loading.
# If you use cuda, you may set it to a greater value like 4 or 8 to accelerate data loading.
num_workers = 8  # You may change this value based on your system configuration.
file_path = "./project_data_food_11" # the location of the dataset
model_path = "./classifier_cnn_v7_batch_128_epoch_160_CutMiX_semi_supervised_AdamW_SGD_Resnet34.pth"


def evaluate_per_class_accuracy(model, valid_loader, device, n_classes=11, class_names=None):
    model.eval()

    correct = torch.zeros(n_classes, dtype=torch.long)
    total   = torch.zeros(n_classes, dtype=torch.long)

    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            for c in range(n_classes):
                mask = (labels == c)
                total[c] += mask.sum().cpu()
                correct[c] += (preds[mask] == c).sum().cpu()

    per_class_acc = correct.float() / total.clamp(min=1).float()
    overall_acc = correct.sum().item() / total.sum().item()

    print(f"\nOverall validation accuracy: {overall_acc:.4f}\n")

    for i, acc in enumerate(per_class_acc):
        name = class_names[i] if class_names is not None else f"Class {i}"
        print(f"{name:15s} | Acc: {acc:.4f} | Samples: {total[i].item()}")

    worst = torch.argsort(per_class_acc)[:3].tolist()
    print("\nWorst performing classes:")
    for i in worst:
        name = class_names[i] if class_names else f"Class {i}"
        print(f"{name:15s} | Acc: {per_class_acc[i]:.4f}")

    return per_class_acc

def calculate_confusion_matrix(model, valid_loader, device):
    model.eval()
    label_true = []
    label_pred = []
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs = imgs.to(device)
            labels = labels.tolist()

            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().tolist()

            label_true.extend(labels)
            label_pred.extend(preds)        
    label_pred = np.array(label_pred)
    label_true = np.array(label_true)
    print("Shape of label_true = {}".format(label_true.shape))
    print("Shape of label_pred = {}".format(label_pred.shape))
    return confusion_matrix(label_true, label_pred) 
    


train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    # transforms.Resize((256, 256)),
    transforms.Resize((142, 142)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    
    # transforms.RandomChoice(
    #     [transforms.AutoAugment(),
    #     transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    #     transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN)]
    # ),
    # transforms.RandomResizedCrop((224, 224), scale=(0.75, 1)),
    # transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ColorJitter(0.2, 0.2),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop((128, 128), scale=(0.3, 1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the image with mean and std of ImageNet dataset.
])

test_tfm = transforms.Compose([
    transforms.Resize((142, 142)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the image with mean and std of ImageNet dataset.
])

train_set = DatasetFolder(file_path + "/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder(file_path + "/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder(file_path + "/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
# unlabeled_set = DatasetFolder(file_path + "/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
test_set = DatasetFolder(file_path + "/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Automatically choose the device to use.
if allow_device:
# Move the network to the appropriate device
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
        print("Using GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load the pretrained classifier
model = Classifier_Resnet18()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint["classifier_network"])
model.to(device)

evaluate_per_class_accuracy(model, valid_loader, device)
model_confusion_matrix = calculate_confusion_matrix(model, valid_loader, device)
print(model_confusion_matrix)
print("May the force be with you!")