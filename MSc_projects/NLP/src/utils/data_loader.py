import os 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import config 
import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split
from ast import literal_eval
from models import model_configurations
# from urllib import request

MAP = {'Unbalanced_power_relations': 0, 
       'Shallow_solution': 1, 
       'Presupposition': 2, 
       'Authority_voice': 3, 
       'Metaphors': 4, 
       'Compassion': 5,
       'The_poorer_the_merrier': 6}

"""
From https://aclanthology.org/2020.coling-main.518.pdf :
For the experimental analysis presented in this paper, we treated paragraphs with final labels 0 and 1 
as negative examples (i.e. as instances not containing PCL) and paragraphs with final labels 2, 3 and 4 
as positive examples (i.e. as instances containing PCL). In total, interpreted in this way, the dataset 
contains 995 positive examples of PCL.
"""
FINAL_LABEL_MAP = {
    0: 0, # no pcl
    1: 0, 
    2: 1, # pcl
    3: 1,
    4: 1
}


class PCLDatasetLoader(Dataset):
    def __init__(self, 
                 file_path: str = config.TRAIN_PATH, 
                 test_ratio: float = config.TRAIN_TEST_SPLIT,
                 model_config = model_configurations.BertConfig):
        
        self.model_config = model_config
        self.tokenizer = model_config.tokenizer
        self.file_path = file_path
        self.test_ratio = test_ratio
        self.data = self._load_dataset()

        if test_ratio is not None:
            self.train, self.dev = self._split_dataset(test_ratio)

    def __len__(self):
        if self.test_ratio is not None:
            return len(self.train), len(self.dev)
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.test_ratio is not None:
            samples = [{
                'texts': str(data['texts'].iloc[index]), 
                'labels': data['labels'].iloc[index]
            } for data in [self.train, self.dev]]
        else:
            samples = {
                'texts': str(self.data['texts'].iloc[index]), 
                'labels': self.data['labels'].iloc[index]
            }
        return samples
    
    def _load_dataset(self) -> pd.DataFrame:
        try:
            if 'tsv' in self.file_path:
                dataset = self._load_tsv()
            else:
                dataset = self._load_csv()
            return dataset
        except FileNotFoundError:
            print(f"Error: File not found at path '{self.file_path}'")
            return None
        except Exception as e:
            print(f"An error occurred while loading the dataset: {e}")
            return None
        
    def _load_tsv(self):
        if self.file_path == '../data/dontpatronizeme_pcl.tsv':
            skip_rows = 4
        else:
            skip_rows = None
        return pd.read_csv(self.file_path, delimiter='\t', header=None,
                            names=['par_id', 'code', 'topic', 'country', 'texts', 'labels'],
                            skiprows=skip_rows)
    def _load_csv(self):
        return pd.read_csv(self.file_path, converters={'labels': literal_eval})
        
    def _split_dataset(self, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_data, test_data = train_test_split(
            self.data, test_size=test_ratio, random_state=config.RS
        )
        return train_data, test_data
    
    def get_datasets(self) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        if self.test_ratio is not None:
            return self.train, self.dev
        else:
            return self.data
    


class PCLDataHandler:
    def __init__(self, model_config=model_configurations.BertConfig, transform=None, balance_ratio=config.BALANCE_RATIO, augment_test=False):
        self.model_config = model_config
        self.transform = transform
        self.balance_ratio = balance_ratio
        self._load_datasets()
        self._build_datasets()
        self.augment_test = augment_test

    def _load_datasets(self):
        self.dpm = PCLDatasetLoader(file_path=config.DATA_PATH,
                                    test_ratio = None,
                                    model_config = self.model_config).get_datasets()

        self.train_dataset, self.dev_dataset = PCLDatasetLoader(file_path=config.TRAIN_PATH, 
                                                                test_ratio = config.TRAIN_TEST_SPLIT,
                                                                model_config = self.model_config).get_datasets()
        
        self.test_dataset = PCLDatasetLoader(file_path=config.DEV_PATH,
                                             test_ratio = None,
                                             model_config = self.model_config).get_datasets()
    
    def _build_datasets(self):
        # add filtering of label=1?
        columns = ['texts', 'labels'] #, 'topic', 'country']
        train_dataset = self._merge_and_filter(self.train_dataset, self.dpm, columns)
        self.train_dataset = self._balance_dataset(train_dataset, balance=self.balance_ratio)
        self.dev_dataset = self._merge_and_filter(self.dev_dataset, self.dpm, columns)
        self.test_dataset = self._merge_and_filter(self.test_dataset, self.dpm, columns)

    @staticmethod
    def _balance_dataset(dataset, balance=0.6):
        """balance TRAIN dataset only: equalise ratio of dpm to non-dpm instances."""
        dpm = (dataset['labels'] == 0)
        non_dpm = (dataset['labels'] == 1)
        mask = dpm & (np.random.rand(len(dpm)) < balance)
        balanced_dataset = pd.merge(dataset[mask], dataset[non_dpm], how='outer')
        # print(np.unique(mask, return_counts=True)) # print resulting numbers
        return balanced_dataset

    @staticmethod
    def _merge_and_filter(df1, df2, column_names, on='par_id'):
        data = pd.merge(df1[[on]], df2, on=on, how='inner')
        data = data[column_names]
        data['labels'] = data['labels'].map(FINAL_LABEL_MAP)
        data = data.dropna() # drop empty rows!
        return data

    def get_dataloaders(self):
        return [
            PCLDataset(
                data=data, 
                model_config = self.model_config,
                transform=self.transform if self.augment_test is True or data is not self.test_dataset else None
                ) for data in [self.train_dataset, self.dev_dataset, self.test_dataset]
        ]
    
    def get_datasets(self):
        return self.train_dataset, self.dev_dataset, self.test_dataset


class PCLDataset(Dataset):
    def __init__(self, 
                 data: pd.DataFrame,
                 model_config = model_configurations.BertConfig,
                 transform = None):
        
        self.model_config = model_config
        self.tokenizer = model_config.tokenizer
        self.data = data
        self.transform = transform # transformation pipeline

    def __len__(self):
        return len(self.data) #len(self.data)

    def __getitem__(self, index):

        if isinstance(index, slice):
            samples = []
            
            start = 0 if index.start is None else index.start
            stop = len(self.data) if index.stop is None else index.stop
            step = 1 if index.step is None else index.step

            for i in range(start, stop, step):
                sample = self.data.iloc[i]
                if self.transform:
                    sample = self.augment_dataset(sample)
                samples.append(sample)
            return pd.DataFrame(samples)
        else:
            sample = self.data.iloc[index]

            if self.transform:
                sample = self.augment_dataset(sample)

            return sample

    def augment_dataset(self, sample):
        sample = self.transform(sample)
        return sample

    def collate_fn(self, batch):
        texts = [item['texts'] for item in batch]
        labels = [item['labels'] for item in batch]

        encodings = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.model_config.max_sequence_length  
        )
    
        encodings['labels'] = torch.tensor(labels)

        return encodings



# transform = DataAugmentation...
# example usage
"""
pcl = PCLDatasetLoader(file_path=config.DATA_PATH,
                        test_ratio = None,
                        model_config = model_configurations.BertConfig).get_datasets()
pcl_dataset = PCLDataset(pcl)
data_loader = DataLoader(pcl_dataset, batch_size=64, shuffle=True)

for batch in data_loader:
    texts = batch['texts']
    labels = batch['labels']
    #print(texts, labels)
    

train_loader = DataLoader(
    pcl_dataset.train,  # Change to pcl_dataset.data if you don't have a separate train set
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    collate_fn=pcl_dataset.collate_fn_bert
)
"""


