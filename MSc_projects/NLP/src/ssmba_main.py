from utils.data_loader import PCLDatasetLoader, FINAL_LABEL_MAP
from utils import config
import os
import pandas as pd
from ast import literal_eval
    
ssmba_path = '../../ssmba_data/'

def save_tsv(data_frame, output_file_path):
    data_frame.to_csv(output_file_path, sep='\t', header=None, index=False)

def create_ssmba_data(dpm):
    dpm_filtered = dpm[dpm['labels'].isin([2, 3, 4])]
    labels = dpm_filtered['labels']
    dpm_data = dpm_filtered['texts']
    save_tsv(labels, os.path.join(ssmba_path, 'dpm_labels.tsv'))
    save_tsv(dpm_data, os.path.join(ssmba_path, 'dpm_text.tsv'))

def create_augmented_data(dpm, text_path):
    print(dpm['labels'].map(FINAL_LABEL_MAP).value_counts())
    train = PCLDatasetLoader(file_path=config.TRANSLATED_DATA_PATH)._load_dataset()
    ssmba_text =pd.read_csv(
        text_path,
        delimiter='\t',  # Adjust if necessary
        quoting=3,  # Quote all fields
        escapechar='\\',
        converters={'labels': literal_eval}
    )
    ssmba_df = pd.DataFrame()

    for label in dpm.columns:
        if label == 'par_id':
            column_value = train[label].iloc[0] # to allocate to train
        elif label == 'texts':
            ssmba_df[label] = ssmba_text
            continue
        elif label == 'labels':
            column_value = 2
        else:
            column_value = dpm[label].iloc[0]
        ssmba_df[label] = [column_value] * len(ssmba_text)

    augmented_dpm = pd.concat([dpm, ssmba_df], ignore_index=True)
    print(augmented_dpm['labels'].map(FINAL_LABEL_MAP).value_counts())
    save_tsv(augmented_dpm, config.AUGMENTED_AND_TRANSLATED_DATA_PATH)

if __name__ == '__main__':
    dpm = PCLDatasetLoader(file_path=config.DATA_PATH)._load_dataset()
    dpm.dropna()
    # create_ssmba_data(dpm)
    # run ssmba and append output
    create_augmented_data(dpm, text_path='../../ssmba/ssmba_out')

