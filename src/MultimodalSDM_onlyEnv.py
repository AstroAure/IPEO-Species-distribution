import os
import glob
import torch
import torch.nn as nn
import satlaspretrain_models 
from satlaspretrain_models import Model

from src.ModelUtils import root

# To create and use a new model:
# 1) Copy "MultimodalSDM.py" with a new name (in the same folder)
# 2) Change variable "name" below (sets the folder where the weights are saved)
# 3) Don't change the class name !
# 4) In Lord_of_the_Main.ipynb, only change the imported file "from src.MultimodalSDM" with the new file name

name = 'multimodal_SDM_only-env'

# neural network with 3 NN (for each modality) and a classifier at the end #same architecture as ex7 and ex9
#maybe add one layer in the CNN for time series?
class multimodal_SDM(nn.Module): #heritates from class nn.Module
    
    def __init__(self, dim_NN_env=128, dim_NN_sat=0, dim_NN_timeseries=0): #dimension of output of neural networks
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
        
        # weights_manager = satlaspretrain_models.Weights()
        # self.CNN_sat = weights_manager.get_pretrained_model("Sentinel2_Resnet50_SI_RGB", fpn=True, head=satlaspretrain_models.Head.CLASSIFY, 
        #                                         num_categories=self.dim_NN_sat, device='cpu')

        # self.CNN_timeseries= nn.Sequential(
        #     #R G B NIR with 10 years and 4 seasons= 40 values: 4 channels, length 40
        #     #Like Alexnet but shorter (2 convolutionnal layers) and in 1D
        #     nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        #     nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        #     nn.Flatten(),
        #     nn.Linear(32*10, dim_NN_timeseries), # 32 channels * length 10 (2 poolings of size 2 equivalent to divide /4 the length of the input) ?
        #     nn.ReLU()
        # )

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
        # satellite_patches = x['satellite_patch'].float()
        # landsat_timeseries = x['landsat_timeseries'].float()

        #pass each modality through its NN
        NN_env_out = self.MLP_env(env_variables)
        # NN_sat_out = self.CNN_sat(satellite_patches)[0]
        # NN_time_series_out = self.CNN_timeseries(landsat_timeseries)

        #concatenate the outputs
        # combined = torch.cat((NN_env_out, NN_sat_out, NN_time_series_out), dim=1)
        combined = NN_env_out

        #pass through the classifier
        output = self.classifier(combined)
        return output

### Model saving and loading functions ###        

def load_model(epoch='latest'):
  # epoch can be:
  # - an integer (number of epoch)
  # - 'latest' (last epoch)
  # - 'best' (highest AUC on validation set)
  # if no model found, a fresh model is returned with epoch=0
  model = multimodal_SDM()
  modelStates = glob.glob(f'{root}/cnn_states/{name}/*.pth')
  if len(modelStates) and (epoch == 'latest' or epoch == 'best' or epoch > 0):
    modelNames = [m.replace(f'{root}/cnn_states/{name}/','').replace('.pth', '') for m in modelStates]
    modelEpochs = [int(m.split('_AUC=')[0]) for m in modelNames]
    modelAUCs = [float(m.split('_AUC=')[1]) if '_AUC=' in m else 0.0 for m in modelNames]
    if epoch == 'latest':
      epoch = max(modelEpochs)
    elif epoch == 'best':
      epoch = modelEpochs[modelAUCs.index(max(modelAUCs))]
    stateDict = torch.load(open(glob.glob(f'{root}/cnn_states/{name}/{epoch}_*.pth')[0], 'rb'), map_location='cpu')
    model.load_state_dict(stateDict)
  else:
    # fresh model
    epoch = 0
  return model, epoch

def save_model(model, epoch, auc_val=None):
  os.makedirs(f'{root}/cnn_states/{name}', exist_ok=True)
  filename = f'{epoch}.pth' if auc_val is None else f'{epoch}_AUC={auc_val:.3f}.pth'
  torch.save(model.state_dict(), open(f'{root}/cnn_states/{name}/{filename}', 'wb'))

def delete_model(epoch):
  os.remove(glob.glob(f'{root}/cnn_states/{name}/{epoch}_*.pth')[0])