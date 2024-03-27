import optuna
from utils.train import DPMTrainer
from utils.predictor import DPMPredictor
from utils.data_loader import PCLDataHandler
import utils.config as config
import os
from utils.data_augmentation import (DataAugmentationPipeline, 
                                     RandomDeleteWords,
                                     RandomSwapWords,
                                     TranslateWords,
                                     SynonymReplacement)
from main import load_model

def objective(trial):
    augmentation_pipeline = DataAugmentationPipeline(
        RandomDeleteWords(),
        RandomSwapWords(),
        # TranslateWords(),  
        SynonymReplacement() 
    )
        
    model_config, model, Trainer, _ = load_model(model_name='ROBERTA')
    train_loader, dev_loader, test_loader = PCLDataHandler(model_config=model_config, transform=augmentation_pipeline, balance_ratio=config.BALANCE_RATIO).get_dataloaders()

    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    num_epochs = trial.suggest_int('num_epochs', 1, 10)
    dropout = trial.suggest_categorical('dropout', [0.1,0.2,0.4,0.5])
    leaky_slope = trial.suggest_categorical('leaky_slope', [0.0,0.01,0.05,0.1,0.2])
    early_stopping_patience = trial.suggest_categorical('early_stopping_patience', [None, 5])
    pos_label_weight = trial.suggest_float('label_weights', 0.5, 1.0)
    
    model_config.learning_rate = learning_rate
    model_config.batch_size = batch_size
    model_config.num_epochs = num_epochs
    model_config.dropout = dropout
    model_config.leaky_slope = leaky_slope
    model_config.early_stopping_patience = early_stopping_patience
    model.pos_label_weight = pos_label_weight
    
    model_config.model_name = f'{search_dir}/trial_{trial.number}'
    trainer = DPMTrainer(Trainer, model, model_config, save_as=model_config.model_name)
    trainer.train(train_loader, dev_loader)

    model_config, loaded_model, _, _ = load_model(model_name='ROBERTA', saved_model_path=model_config.model_name)
    predictor = DPMPredictor(model=loaded_model, model_config=model_config)
    predictor.evaluate(test_loader)
    return predictor.get_f1_score()

def run_optuna():
    study = optuna.create_study(direction='maximize', study_name='roBERTa hyperparameter search')
    study.optimize(objective, n_trials=500)  

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == '__main__':
    search_dir = 'models/finetunned/roberta_search'
    os.makedirs(search_dir, exist_ok=True)
    run_optuna()


    

