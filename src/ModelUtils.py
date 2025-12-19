import torch
import torch.nn as nn
from torch.utils.data import dataset
from torch.utils.data import DataLoader
from torch.optim import SGD
import numpy as np
import os
import glob
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from src.GeoPlantDataset import GeoPlantDataset, viz_sample


def load_dataloader(batch_size, split='train', num_workers=1):
  return DataLoader(
      GeoPlantDataset(data_folder=f"{root}/data", split=split),
      batch_size=batch_size,
      shuffle=(split=='train'),       # we shuffle the image order for the training dataset
      num_workers=num_workers                   # perform data loading with one CPU threads
  )


def setup_optimiser(model, learning_rate, weight_decay):
  return SGD(
    model.parameters(),
    learning_rate,
    weight_decay
  )

def train_epoch(data_loader, model, optimiser, criterion=nn.BCELoss(), device='cuda'):
  # set model to training mode. This is important because some layers behave differently during training and testing
  model.train(True)
  model.to(device)
  # stats
  loss_total = 0.0
  # iterate over dataset
  for idx, sample in tqdm(enumerate(data_loader)):
    # put data onto correct device and separate target from data
    sample = {key: value.to(device) for key, value in sample.items()}
    target = sample['species_labels'].float()
    data = {key: value for key, value in sample.items() if key != 'species_labels'}
    # reset gradients
    optimiser.zero_grad()
    # forward pass
    pred = model(data)
    # loss
    loss = criterion(pred, target)
    # backward pass
    loss.backward()
    # parameter update
    optimiser.step()
    # stats update
    loss_total += loss.item()
  # normalise stats
  loss_total /= len(data_loader)
  return model, loss_total

@torch.no_grad()
def validate_epoch(data_loader, model, criterion=nn.BCELoss(), device='cuda'):       # note: no optimiser needed
  # set model to evaluation mode
  model.train(False)
  model.to(device)
  # stats
  loss_total = 0.0
  pred_list = []
  target_list = []
  # iterate over dataset
  for idx, sample in tqdm(enumerate(data_loader)):
    with torch.no_grad():
      # put data onto correct device and separate target from data
      sample = {key: value.to(device) for key, value in sample.items()}
      target = sample['species_labels'].float()
      data = {key: value for key, value in sample.items() if key != 'species_labels'}
      # forward pass
      pred = model(data)
      # loss
      loss = criterion(pred, target)
      # stats update
      loss_total += loss.item()
      pred_list.append(pred.cpu().numpy())
      target_list.append(target.cpu().numpy())
  # AUC calculation
  pred_list = np.concatenate(pred_list, axis=0)
  target_list = np.concatenate(target_list, axis=0)
  fpr=[0 for _ in range(342)]
  tpr=[0 for _ in range(342)]
  roc_auc= [0 for _ in range(342)]
  for i in range (342):
    fpr[i], tpr[i], _ = roc_curve(target_list[:, i], pred_list[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
  auc_total = np.mean(roc_auc)
  # normalise stats
  loss_total /= len(data_loader)
  auc_total /= len(data_loader)
  return loss_total,auc_total
  

# Model size
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


### UTILS

def find_root_dir():
    try:
        root = Path(__file__).resolve().parent
    except NameError:
        root = Path.cwd()  # fallback for Jupyter notebooks

    while root.parent != root:
        if any((root / marker).exists() for marker in ["README.md"]):
            break
        root = root.parent

    # Fallback in case nothing found
    if not any((root / marker).exists() for marker in ["README.md"]):
        print("Could not locate project root — defaulting to current working directory")
        root = Path.cwd()
    root = str(root)
    return root

root = find_root_dir()
