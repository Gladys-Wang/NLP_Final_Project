# much code in this module is borrowed from the Eraser Benchmark (DeYoung et al., 2019)
import os
import sys
import json
import torch
import torch.nn as nn
from os.path import join, dirname, abspath
from transformers import RobertaTokenizer, RobertaForSequenceClassification, PretrainedConfig
from evidence_inference.models.utils import PaddedSequence

# Add the parent directory to the system path to access the necessary modules
sys.path.insert(0, abspath(join(dirname(__file__), '..', '..')))

def initialize_models(params: dict):
    tokenizer = RobertaTokenizer.from_pretrained(params['bert_vocab'])
    pad_token_id, cls_token_id, sep_token_id = tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id
    evidence_classes = {y: x for x, y in enumerate(params['evidence_classifier']['classes'])}

    if params.get('random_init', False):
        id_config, cls_config = load_config(params['bert_config'], len(evidence_classes))
        evidence_identifier, evidence_classifier = create_classifiers(None, pad_token_id, cls_token_id, sep_token_id, params['max_length'], params, id_config, cls_config)
    else:
        evidence_identifier, evidence_classifier = create_classifiers(params['bert_dir'], pad_token_id, cls_token_id, sep_token_id, params['max_length'], params, num_labels=[2, len(evidence_classes)])

    word_interner = tokenizer.get_vocab()
    de_interner = {v: k for k, v in word_interner.items()}
    return evidence_identifier, evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer

def load_config(config_path, num_classes):
    with open(config_path, 'r') as file:
        config_json = json.load(file)
    return (PretrainedConfig.from_dict(config_json, num_labels=2),
            PretrainedConfig.from_dict(config_json, num_labels=num_classes))

def create_classifiers(bert_dir, pad_token_id, cls_token_id, sep_token_id, max_length, params, id_config=None, cls_config=None):
    num_labels = [2, len(id_config.num_labels)]
    classifiers = []
    for config in [id_config, cls_config]:
        classifier = BertClassifier(
            bert_dir, pad_token_id, cls_token_id, sep_token_id, num_labels.pop(0), max_length,
            use_half_precision=params['use_half_precision'], config=config)
        classifiers.append(classifier)
    return classifiers

class BertClassifier(nn.Module):
    def __init__(self, bert_dir: str, pad_token_id: int, cls_token_id: int, sep_token_id: int, num_labels: int, max_length: int=512, use_half_precision: bool=False, config: PretrainedConfig=None):
        super().__init__()
        self.bert = self.initialize_bert(bert_dir, config, num_labels)
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.max_length = max_length
        if use_half_precision:
            from apex import amp
            self.bert = amp.initialize(self.bert, opt_level="O1")

    def initialize_bert(self, bert_dir, config, num_labels):
        if bert_dir:
            return RobertaForSequenceClassification.from_pretrained(bert_dir, num_labels=num_labels)
        else:
            return RobertaForSequenceClassification(config)

    def forward(self, query: torch.Tensor, document_batch: torch.Tensor):
        device = next(self.bert.parameters()).device
        cls_token = torch.tensor([self.cls_token_id], device=device)
        sep_token = torch.tensor([self.sep_token_id], device=device)

        input_tensors = [torch.cat([cls_token, q, sep_token, d[:self.max_length - len(q) - 2]]) for q, d in zip(query, document_batch)]
        bert_input = PaddedSequence.autopad(input_tensors, batch_first=True, padding_value=self.pad_token_id, device=device)
        
        outputs = self.bert(input_tensor=bert_input.data, attention_mask=bert_input.mask(on=1.0, off=0.0, dtype=torch.float, device=device))
        return outputs.logits


