
from transformers import BertTokenizer, RobertaTokenizer
import torch
import os

FINE_TUNNED_DIR = 'models/finetunned/'
os.makedirs(FINE_TUNNED_DIR, exist_ok=True)

class RandomTokenizer:
    def __init__(self):
        pass

    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return ''.join([chr(token_id) for token_id in token_ids])

class BertConfig(object):
    model_name = os.path.join(FINE_TUNNED_DIR, 'bert_with_synonym_replacement')
    save_path = None
    log_path = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 3
    batch_size = 32
    learning_rate = 5e-5
    bert_path = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    hidden_size = 768
    num_classes = 2
    max_sequence_length = 128 # max sequence length for bert is 512
    early_stopping_patience = None #10 # set to None if not

    def __init__(self, dataset, pretrained_name_or_path=None):
        self.save_path = dataset + '/saved_models/' + self.model_name + '.ckpt'  
        self.log_path = dataset + '/log/' + self.model_name
        self.bert_path = pretrained_name_or_path or self.bert_path
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path) 
    
class RandomConfig(object):
    model_name = os.path.join(FINE_TUNNED_DIR, 'random_model')
    save_path = None
    log_path = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 3
    batch_size = 32
    learning_rate = 5e-5
    num_classes = 2
    tokenizer = RandomTokenizer()
    early_stopping_patience = None

class RobertaConfig(object):
    model_name = os.path.join(FINE_TUNNED_DIR, 'roberta_model')
    save_path = None
    log_path = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 5
    batch_size = 32
    learning_rate = 4.0928349288335714e-05
    roberta_path = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
    hidden_size = 768
    num_classes = 2
    leaky_slope = 0.1
    dropout = 0.2
    max_sequence_length = 128 # max sequence length for bert is 512
    early_stopping_patience = None # 10 # set to None if not
    pos_label_weight = 0.6226696637246095