from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

SPLITS = {'train': 'training', 
          'test': 'test', 
          'val': None}
# TODO: Separate 'training' data into 'train' and 'val' sets

class GeoPlantDataset(Dataset):
    def __init__(self, data_folder, transform=None, split='train'):
        self.data_folder = data_folder
        # Load data
        fold = SPLITS[split]
        self.env_variables = pd.read_csv(f'{data_folder}/env_variables_{fold}.csv')
        self.landsat_timeseries = pd.read_csv(f'{data_folder}/landsat_timeseries_{fold}.csv')
        self.satellite_patches = np.load(f'{data_folder}/satellite_patches_{fold}.npy')
        self.species_data = np.load(f'{data_folder}/species_data_{fold}.npy')
        # Extract relevant columns
        self.survey_ids = self.env_variables['surveyId']
        self.lon = self.env_variables['lon'].values
        self.lat = self.env_variables['lat'].values
        # Store transforms
        self.transform = transform

    def __len__(self):
        return len(self.env_variables)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Survey ID associated with the index
        survey_id = self.survey_ids.iloc[idx]
        # Extract data components
        env = self.env_variables.iloc[idx].drop(['surveyId', 'lon', 'lat']).values.astype(np.float32) # Shape: (num_env_variables,)
        timeseries = self.landsat_timeseries.iloc[idx].drop(['surveyId']).values.astype(np.float32) # Shape: (num_bands * num_time_steps,)
        patch = self.satellite_patches[idx].astype(np.float32) # Shape: (bands, height, width)
        species = self.species_data[idx].astype(np.float32) # Shape: (num_species,)
        # Package sample
        sample = {'survey_id': survey_id,
                  'env_variables': env,
                  'landsat_timeseries': timeseries,
                  'satellite_patch': patch,
                  'species_labels': species}

        # Apply transformations if any
        if self.transform:
            sample = self.transform(sample)

        return sample