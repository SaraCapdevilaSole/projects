import torch
from transformers import TrainingArguments, IntervalStrategy, EarlyStoppingCallback
import tqdm as tqdm

"""
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def compute_metrics(p):    
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
"""

class DPMTrainer:
    def __init__(self, Trainer, model, model_config, save_as=None):
        self.model = model
        self.model_config = model_config
        self.Trainer = Trainer
        if save_as is None:
            save_as = model_config.model_name
        self.save_as = save_as

    def train(self, train_loader, dev_loader):
        training_args = TrainingArguments(
            disable_tqdm=False,
            output_dir=self.save_as,
            learning_rate=self.model_config.learning_rate,
            logging_steps=100,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=100,
            # prediction_loss_only=True,
            save_strategy=IntervalStrategy.NO,
            per_device_train_batch_size=self.model_config.batch_size,
            per_device_eval_batch_size=self.model_config.batch_size,
            num_train_epochs=self.model_config.num_epochs,
            # metric_for_best_model='f1',
            # load_best_model_at_end=True,
        )

        callbacks = self._gather_callbacks()

        trainer = self.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_loader,
            eval_dataset=dev_loader,
            data_collator=train_loader.collate_fn,
            # compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        torch.cuda.memory_stats_as_nested_dict()
        trainer.train()
        trainer.save_model(self.save_as)

    def _gather_callbacks(self):
        callbacks = []
        if self.model_config.early_stopping_patience is not None:
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=self.model_config.early_stopping_patience
            )
            callbacks.append(early_stopping_callback)
        return callbacks
