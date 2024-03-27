import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from models import model_configurations
from utils.data_loader import PCLDatasetLoader
import pandas as pd
import os

from typing import List, Dict, Tuple

class DPMPredictor:
    def __init__(self, model, model_config=model_configurations.BertConfig, is_random: bool = False):
        self.model = model  # loaded model
        self.report: Dict[str, Dict[str, float]] = {}  # initialise report
        self.is_random = is_random
        if not is_random:
            self.tokenizer = model_config.tokenizer
            self.model.eval()

    def evaluate(self, data_loader) -> Dict[str, Dict[str, float]]:
        tot_labels, preds = self._run_evaluation(data_loader)
        report = classification_report(tot_labels, preds, target_names=["Not_Patronising", "Patronising"], output_dict=True)
        self.report = report
        return report
    
    def get_f1_score(self, label: str = 'Patronising') -> float:
        for report_label, metrics in self.report.items():
            if report_label.lower() == label.lower():
                if isinstance(metrics, dict) and 'f1-score' in metrics:
                    return metrics['f1-score']
        return 0.0

    def _run_evaluation(self, data_loader, output_csv_path: str = '../data/output.csv') -> Tuple[List[int], List[int]]:
        preds = []
        true_labels = []
        texts_list = []
        with torch.no_grad():
            for data in tqdm(data_loader):
                texts = data['texts']
                labels = data['labels']

                if self.is_random:
                    model = self.model()
                    output = model(data)
                    predicted_class = output[0]
                else:
                    encodings = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
                    output = self.model(**encodings)
                    _, predicted_class = torch.max(output, 1)

                preds.append(predicted_class.tolist())
                true_labels.append(labels.tolist())
                texts_list.append(texts)

        # save official dev set predictions
        with open('../data/dev.txt', 'w') as file:
            for p in preds:
                file.write(f"{p[0]}\n")

        # save for analysis 
        df = pd.DataFrame({'texts': texts_list, 'labels': true_labels, 'preds': preds})
        df.to_csv(output_csv_path, index=False)
        return true_labels, preds
    
    def predict_official_test(self, output_file: str = '../data/test.txt'):
        data = PCLDatasetLoader('../data/task4_test.tsv')._load_dataset()
        preds = []

        print('Predicting official test file:\n')
        with torch.no_grad():
            for _, row in tqdm(data.iterrows()):
                texts = row['texts']

                encodings = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
                output = self.model(**encodings)
                _, predicted_class = torch.max(output, 1)

                preds.append(predicted_class.tolist())

        with open(output_file, 'w') as file:
            for p in preds:
                file.write(f"{p[0]}\n")
    
    def print_report(self) -> None:
        for label, metrics in self.report.items():
            print(f"\nMetrics for {label}:")
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    print(f"{metric}: {value}")
            else:
                print(f"Value: {metrics}")


