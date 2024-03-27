import torch
import torch.nn as nn
from typing import List, Optional
import logging
from transformers import Trainer, RobertaModel, RobertaPreTrainedModel, logging
from models.model_configurations import RobertaConfig

class Model(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.residual = nn.Linear(config.hidden_size, config.hidden_size//2)

        self.projection = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.BatchNorm1d(config.hidden_size),
            torch.nn.LeakyReLU(negative_slope=RobertaConfig.leaky_slope),
            torch.nn.Dropout(RobertaConfig.dropout),
            torch.nn.Linear(config.hidden_size, config.hidden_size//2),
            torch.nn.BatchNorm1d(config.hidden_size//2),
            torch.nn.LeakyReLU(negative_slope=RobertaConfig.leaky_slope),
            torch.nn.Dropout(RobertaConfig.dropout),
        )

        self.out_layer = torch.nn.Linear(config.hidden_size//2, RobertaConfig.num_classes)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.out_layer(self.projection(outputs[1]) + self.residual(outputs[1]))
        return logits


class Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_task = nn.CrossEntropyLoss().to(device) # weight=torch.FloatTensor([RobertaConfig.pos_label_weight, 1 - RobertaConfig.pos_label_weight]
        loss = loss_task(outputs.view(-1, 2), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
    
