from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SPLITS = {'train': 'train',  # 4000 samples
          'val': 'val',      # 1000 samples
          'test': 'test'     # 1000 samples
          }

# From https://www.worldclim.org/data/bioclim.html
ENV_VARS = ['BIO1 (Annual Mean Temperature)',
            'BIO2 (Mean Diurnal Range (Mean of monthly (max temp - min temp)))',
            'BIO3 (Isothermality (BIO2/BIO7) (x100))',
            'BIO4 (Temperature Seasonality (standard deviation x100))',
            'BIO5 (Max Temperature of Warmest Month)',
            'BIO6 (Min Temperature of Coldest Month)',
            'BIO7 (Temperature Annual Range (BIO5-BIO6))',
            'BIO8 (Mean Temperature of Wettest Quarter)',
            'BIO9 (Mean Temperature of Driest Quarter)',
            'BIO10 (Mean Temperature of Warmest Quarter)',
            'BIO11 (Mean Temperature of Coldest Quarter)',
            'BIO12 (Annual Precipitation)',
            'BIO13 (Precipitation of Wettest Month)',
            'BIO14 (Precipitation of Driest Month)',
            'BIO15 (Precipitation Seasonality (Coefficient of Variation))',
            'BIO16 (Precipitation of Wettest Quarter)',
            'BIO17 (Precipitation of Driest Quarter)',
            'BIO18 (Precipitation of Warmest Quarter)',
            'BIO19 (Precipitation of Coldest Quarter)']

TIMESERIES_BANDS = ['Red', 'Green', 'Blue', 'NIR']
TIMESERIES_YEARS = np.arange(2008, 2018)
TIMESERIES_QUARTERS = ['Q1', 'Q2', 'Q3', 'Q4']
TIMESERIES_COLORS = {'Red': 'r', 
                     'Green': 'g', 
                     'Blue': 'b', 
                     'NIR': 'maroon'}

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
        '''
        Retrieve a sample from the dataset at the given index.
        Returns a dictionary with the following keys:
        - 'survey_id': Survey ID
        - 'lon': Longitude
        - 'lat': Latitude
        - 'env_variables': Environmental variables (np.array : (19,))
        - 'landsat_timeseries': Landsat time series data (np.array: (4, 40))
        - 'satellite_patch': Satellite image patch (np.array: (3, 128, 128))
        - 'species_labels': Species presence/absence labels (np.array: (342,))
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Survey ID associated with the index
        survey_id = self.survey_ids.iloc[idx]
        # Extract data components
        env = self.env_variables.iloc[idx].drop(['surveyId', 'lon', 'lat']).values.astype(np.int32) # Shape: (num_env_variables,)
        timeseries = self.landsat_timeseries.iloc[idx].drop(['surveyId']).values.astype(np.int32).reshape(len(TIMESERIES_BANDS), -1) # Shape: (num_bands, num_time_steps)
        patch = self.satellite_patches[idx].astype(np.int32) # Shape: (bands, height, width)
        species = self.species_data[idx].astype(np.int32) # Shape: (num_species,)
        # Package sample
        sample = {'survey_id': survey_id,
                  'lon': self.lon[idx],
                  'lat': self.lat[idx],
                  'env_variables': env,
                  'landsat_timeseries': timeseries,
                  'satellite_patch': patch,
                  'species_labels': species}

        # Apply transformations if any
        if self.transform:
            sample = self.transform(sample)

        return sample

def viz_env(env):
    '''Visualize environmental variables.'''
    for i, var in enumerate(ENV_VARS):
        print(f"{var:.<66}: {env[i]}")

def viz_timeseries(timeseries):
    '''Visualize Landsat time series data.'''
    fig, ax = plt.subplots(figsize=(10, 3))
    len_data = len(TIMESERIES_YEARS)*len(TIMESERIES_QUARTERS)
    for i, band in enumerate(TIMESERIES_BANDS):
        band_data = timeseries[i]
        ax.plot(np.arange(len(band_data)), band_data, marker='D', color=TIMESERIES_COLORS[band], label=band)
    ax.set_xticks(np.arange(len_data, step=4))
    ax.set_xticklabels([f"{year}-Q1" for year in TIMESERIES_YEARS], rotation=45)
    ax.set_ylabel("Mean reflectance [A.U.]")
    ax.set_title("Landsat time series")
    ax.legend(loc='center right')
    plt.show()

def viz_patch(patch):
    '''Visualize satellite image patch.'''
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(np.transpose(patch, (1, 2, 0)))
    ax.set_title("Sentinel-2 image patch (10m/px)")
    ax.axis('off')
    plt.show()

def viz_species(species):
    '''Visualize species labels.'''
    print(species)

def viz_sample(sample):
    '''Visualize all components of a dataset sample.'''
    print(f"Survey ID: {sample['survey_id']}")
    print(f"Location: ({sample['lat']}, {sample['lon']})")
    print("== Environmental Variables ==")
    viz_env(sample['env_variables'])
    print("== Landsat Timeseries ==")
    viz_timeseries(sample['landsat_timeseries'])
    print("== Satellite Patch ==")
    viz_patch(sample['satellite_patch'])
    print("== Species Labels ==")
    viz_species(sample['species_labels'])