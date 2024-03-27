import translators as ts
import random
from utils.data_loader import PCLDatasetLoader
from utils import config
from tqdm import tqdm
from utils.data_augmentation import TranslateWords
    
def save_tsv(data_frame, output_file_path):
    data_frame.to_csv(output_file_path, sep='\t', header=None, index=False)

if __name__ == '__main__':
    dpm = PCLDatasetLoader(file_path=config.DATA_PATH)._load_tsv()
    dpm.dropna()
    tqdm.pandas()
    translate = TranslateWords()
    translated = dpm.progress_apply(lambda row: translate(row['texts']), axis=1)
    dpm['texts'] = translated
    save_tsv(dpm, config.TRANSLATED_DATA_PATH)