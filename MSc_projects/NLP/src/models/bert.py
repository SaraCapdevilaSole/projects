import torch
import torch.nn as nn
from typing import List, Optional
import logging
from transformers import Trainer, BertModel, BertPreTrainedModel, logging
from models.model_configurations import BertConfig
from models.bert import BertModel
from typing import Dict

logging.set_verbosity_error()


# models: Bert

class Model(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        #self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size // 2),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(config.hidden_size // 2, BertConfig.num_classes),
        )

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
        outputs = self.bert(
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

        # Logits
        logits = self.projection(outputs[1])

        return logits


class Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # logs: Dict[str, float] = {}
        # self.log.extend(logs)

        labels = inputs.pop('labels')
        outputs = model(**inputs)

        loss_task = nn.CrossEntropyLoss()
        loss = loss_task(outputs.view(-1, 2), labels.view(-1))

        return (loss, outputs) if return_outputs else loss
    

