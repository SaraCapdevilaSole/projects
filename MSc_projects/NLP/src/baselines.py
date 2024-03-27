from utils.data_loader import PCLDataHandler
import utils.config as config
import tqdm as tqdm
from utils.data_cleaning import *

import argparse
from sklearn.feature_extraction.text import (TfidfVectorizer, 
                                             CountVectorizer)
from sklearn.naive_bayes import (GaussianNB,
                                 MultinomialNB)
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


parser = argparse.ArgumentParser(description="Train and test a baseline model.")
parser.add_argument("--vectorizer", choices=["BagOfWords", "BagOfBigrams", "BagOfTrigrams", "TfIdf"], default="TfIdf")
parser.add_argument("--model", choices=["LogisticRegression", "SVM", "NaiveBayes"], default="LogisticRegression")
parser.add_argument("--oversample", choices=["True", "False"], default="True")
args = parser.parse_args()

vectorizer = None
if args.vectorizer == "BagOfWords":
    vectorizer = CountVectorizer()
elif args.vectorizer == "BagOfBigrams":
    vectorizer = CountVectorizer(ngram_range=(2,2))
elif args.vectorizer == "BagOfTrigrams":
    vectorizer = CountVectorizer(ngram_range=(3,3))
elif args.vectorizer == "TfIdf":
    vectorizer = TfidfVectorizer()

model = None
if args.model == "LogisticRegression":
    model = LogisticRegression()
elif args.model == "SVM":
    model = SVC()
elif args.model == "NaiveBayes":
    model = MultinomialNB()

cleaning_pipeline = DataCleaningPipeline(TokenizeLinks(),
                                         RemoveReferencing(),
                                         RemoveSpecialCharaters(),
                                         RemoveStopwords(),
                                         RemoveShortWords(),
                                         StemWords())


train_loader, dev_loader, test_loader = PCLDataHandler(transform=cleaning_pipeline, augment_test=True).get_dataloaders()

vectorizer.fit(train_loader[:]['texts'])

train_vec_texts = vectorizer.transform(train_loader[:]['texts'])
train_vec_labels = train_loader[:]['labels']
test_vec_texts = vectorizer.transform(test_loader[:]['texts'])
test_vec_labels = test_loader[:]['labels']

if args.oversample:
    smote = SMOTE()
    train_vec_texts, train_vec_labels = smote.fit_resample(train_vec_texts, train_vec_labels)

model.fit(train_vec_texts, train_vec_labels)
pred = model.predict(test_vec_texts)

report = classification_report(test_vec_labels, pred, target_names=["Not_Patronising", "Patronising"], output_dict=True)
for label, metrics in report.items():
    print(f"\nMetrics for {label}:")
    if isinstance(metrics, dict):
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    else:
        print(f"Value: {metrics}")