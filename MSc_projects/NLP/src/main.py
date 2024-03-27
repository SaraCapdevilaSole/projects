import torch
import transformers
from transformers import BertTokenizer, TrainingArguments, IntervalStrategy, EarlyStoppingCallback, TrainerCallback
from models import bert, random, roberta
from utils.data_loader import PCLDataHandler
import utils.config as config
from models import model_configurations
import tqdm as tqdm
from utils.train import DPMTrainer
import argparse
from utils.data_augmentation import (DataAugmentationPipeline, 
                                     RandomDeleteWords,
                                     RandomSubstitution,
                                     RandomSwapWords,
                                     TranslateWords,
                                     SynonymReplacement)
from utils.predictor import DPMPredictor

def load_model(model_name: str, saved_model_path=None) -> transformers:
    is_random = False
    if model_name == "BERT":
        model_config = model_configurations.BertConfig
        if saved_model_path:
            # to load a saved model - for testing
            model = bert.Model.from_pretrained(saved_model_path)
        else:
            # to finetunne existing model during training
            model = bert.Model.from_pretrained(model_configurations.BertConfig.bert_path)
        Trainer = bert.Trainer
    if model_name == "RANDOM": 
        model_config = model_configurations.RandomConfig
        model = random.Model
        Trainer = None
        is_random = True
    if model_name == "ROBERTA":
        model_config = model_configurations.RobertaConfig
        if saved_model_path:
            model = roberta.Model.from_pretrained(saved_model_path)
        else:
            model = roberta.Model.from_pretrained(model_configurations.RobertaConfig.roberta_path)
        Trainer = roberta.Trainer
    return model_config, model, Trainer, is_random

def main():
    parser = argparse.ArgumentParser(description="Train or test the model.")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Specify whether to train or test the model.")
    parser.add_argument("--model_name", type=str, default=config.MODEL, help="Specify the model to use.")
    parser.add_argument("--save_as", type=str, default=None, help="Specify the name for the saved model.")
    args = parser.parse_args()

    model_config, model, Trainer, _ = load_model(model_name=args.model_name)

    if args.mode == 'train':
        assert Trainer is not None, "Cannot train without a Trainer! \n - Note: Perform Inferenece directly if using the Random Model"

    save_n_load_name = args.save_as if args.save_as is not None else model_config.model_name

    augmentation_pipeline = DataAugmentationPipeline(
        RandomDeleteWords(),
        RandomSwapWords(), 
        # TranslateWords(),  # Use translated dataset instead
        SynonymReplacement(), 
        RandomSubstitution()
    )
    
    # Augmentation on Test is None
    train_loader, dev_loader, test_loader = PCLDataHandler(model_config=model_config, transform=augmentation_pipeline, balance_ratio=config.BALANCE_RATIO).get_dataloaders()

    if args.mode == "train":
        trainer = DPMTrainer(Trainer, model, model_config, save_as=save_n_load_name)
        trainer.train(train_loader, dev_loader)

    elif args.mode == "test":
        model_config, loaded_model, _, is_random = load_model(model_name=args.model_name, saved_model_path=save_n_load_name)
        
        predictor = DPMPredictor(model=loaded_model, model_config=model_config, is_random=is_random)
        predictor.evaluate(test_loader)
        predictor.print_report()

        # predict official test set labels
        # predictor.predict_official_test()

if __name__ == '__main__':
    main()


    

