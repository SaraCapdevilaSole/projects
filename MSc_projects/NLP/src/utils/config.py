import torch
import os

RS = 22  # random state

MODEL = 'BERT'
TRAIN_TEST_SPLIT = 0.2
BALANCE_RATIO = 1.0

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    torch.cuda.empty_cache() 
else:
    DEVICE = 'cpu'

DATA = '../data/'
DATA_PATH = os.path.join(DATA, 'dontpatronizeme_pcl.tsv')
TRAIN_PATH = os.path.join(DATA, 'train_semeval_parids-labels.csv')
DEV_PATH = os.path.join(DATA, 'dev_semeval_parids-labels.csv')
TRANSLATED_DATA_PATH = os.path.join(DATA, 'dontpatronizeme_pcl_translated.tsv')
AUGMENTED_DATA_PATH = os.path.join(DATA, 'dontpatronizeme_pcl_augmented.tsv')
AUGMENTED_AND_TRANSLATED_DATA_PATH = os.path.join(DATA, 'dontpatronizeme_pcl_augmented_translated.tsv')

DATA_PATH = AUGMENTED_AND_TRANSLATED_DATA_PATH