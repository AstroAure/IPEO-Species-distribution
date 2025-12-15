import torch
import torch.nn as nn
import satlaspretrain_models 
from satlaspretrain_models import Model
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

from GeoPlantDataset import GeoPlantDataset, viz_sample


def load_dataloader(batch_size, split='train', num_workers=1):
  return DataLoader(
      GeoPlantDataset(data_folder=f"{root}/data", split=split),
      batch_size=batch_size,
      shuffle=(split=='train'),       # we shuffle the image order for the training dataset
      num_workers=num_workers                   # perform data loading with one CPU threads
  )


# neural network with 3 NN (for each modality) and a classifier at the end #same architecture as ex7 and ex9
#maybe add one layer in the CNN for time series?
class multimodal_SDM (nn.Module): #heritates from class nn.Module
    
    def __init__(self, dim_NN_env=128, dim_NN_sat=256, dim_NN_timeseries=64): #dimension of output of neural networks
        super(multimodal_SDM, self).__init__()  #call the init of the parent class
        
        self.dim_NN_env=dim_NN_env

        self.dim_NN_sat=dim_NN_sat

        self.dim_NN_timeseries=dim_NN_timeseries
        
        self.MLP_env = nn.Sequential( #19 bioclim variables to 128 values
            nn.Linear(19,50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, dim_NN_env),
            nn.ReLU()
            )
        
        weights_manager = satlaspretrain_models.Weights()
        self.CNN_sat = weights_manager.get_pretrained_model("Sentinel2_Resnet50_SI_RGB", fpn=True, head=satlaspretrain_models.Head.CLASSIFY, 
                                                num_categories=self.dim_NN_sat, device='cpu')
        # self.CNN_sat = Model(weights=torch.load("Aurelien_DataLoader/sentinel2_resnet50_si_rgb.pth", map_location=torch.device('cpu')), 
        #                      backbone=satlaspretrain_models.Backbone.RESNET50, 
        #                      fpn=True, head=satlaspretrain_models.Head.CLASSIFY, num_categories=self.dim_NN_sat)


        self.CNN_timeseries= nn.Sequential(
            #R G B NIR with 10 years and 4 seasons= 40 values: 4 channels, length 40
            #Like Alexnet but shorter (2 convolutionnal layers) and in 1D
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32*10, dim_NN_timeseries), # 32 channels * length 10 (2 poolings of size 2 equivalent to divide /4 the length of the input) ?
            nn.ReLU()
        )

        self.classifier= nn.Sequential(
            nn.Linear(dim_NN_env+dim_NN_sat+dim_NN_timeseries, 1000),
            nn.ReLU(),
            nn.Linear(1000, 342),
            nn.Sigmoid()
        )

    def forward (self, x):
        '''x is a GeoPlantDataset object containing the 3 modalities:
        - env_variables
        - satellite_patch
        - landsat_timeseries
        '''
        env_variables = x['env_variables'].float()
        satellite_patches = x['satellite_patch'].float()
        landsat_timeseries = x['landsat_timeseries'].float()

        #pass each modality through its NN
        NN_env_out = self.MLP_env(env_variables)
        NN_sat_out = self.CNN_sat(satellite_patches)[0]
        NN_time_series_out = self.CNN_timeseries(landsat_timeseries)

        #concatenate the outputs
        combined = torch.cat((NN_env_out, NN_sat_out, NN_time_series_out), dim=1)

        #pass through the classifier
        output = self.classifier(combined)
        return output


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

def load_model(epoch='latest'):
  model = multimodal_SDM()
  modelStates = glob.glob(f'{root}/cnn_states/multimodal_SDM/*.pth')
  if len(modelStates) and (epoch == 'latest' or epoch > 0):
    modelStates = [int(m.replace(f'{root}/cnn_states/multimodal_SDM/','').replace('.pth', '')) for m in modelStates]
    if epoch == 'latest':
      epoch = max(modelStates)
    stateDict = torch.load(open(f'{root}/cnn_states/multimodal_SDM/{epoch}.pth', 'rb'), map_location='cpu')
    model.load_state_dict(stateDict)
  else:
    # fresh model
    epoch = 0
  return model, epoch


def save_model(model, epoch):
  os.makedirs(f'{root}/cnn_states/multimodal_SDM', exist_ok=True)
  torch.save(model.state_dict(), open(f'{root}/cnn_states/multimodal_SDM/{epoch}.pth', 'wb'))

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
