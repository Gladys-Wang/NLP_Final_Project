# much code in this module is borrowed from the Eraser Benchmark (DeYoung et al., 2019)
from os.path import join, dirname, abspath
import sys
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertForSequenceClassification, PretrainedConfig
import adapters
from evidence_inference.models.utils import PaddedSequence

# Adjust the system path to include the module directory
sys.path.insert(0, abspath(join(dirname(__file__), '..', '..')))

def initialize_models(params: dict):
    max_length = params['max_length']
    tokenizer = AutoTokenizer.from_pretrained("gsarti/biobert-nli", ignore_mismatched_sizes=True)
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    evidence_classes = {y: x for x, y in enumerate(params['evidence_classifier']['classes'])}

    config_options = {
        'pad_token_id': pad_token_id,
        'cls_token_id': cls_token_id,
        'sep_token_id': sep_token_id,
        'max_length': max_length,
        'use_half_precision': params.get('use_half_precision', False)
    }

    if params.get('random_init', False):
        with open(params['bert_config'], 'r') as file:
            config_data = json.load(file)
        config = PretrainedConfig.from_dict(config_data, num_labels=len(evidence_classes))
        evidence_identifier = BertClassifier(num_labels=2, config=config, **config_options)
        evidence_classifier = BertClassifier(num_labels=len(evidence_classes), config=config, **config_options)
    else:
        bert_dir = params['bert_dir']
        evidence_identifier = BertClassifier(bert_dir=bert_dir, num_labels=2, **config_options)
        evidence_classifier = BertClassifier(bert_dir=bert_dir, num_labels=len(evidence_classes), **config_options)

    word_interner = tokenizer.get_vocab()
    de_interner = {v: k for k, v in word_interner.items()}
    return evidence_identifier, evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer

class BertClassifier(nn.Module):
    def __init__(self, bert_dir: Optional[str], num_labels: int, pad_token_id: int, cls_token_id: int, sep_token_id: int, max_length: int=512, use_half_precision=False, config: Optional[PretrainedConfig]=None):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(bert_dir, num_labels=num_labels) if bert_dir else BertForSequenceClassification(config)
        adapters.init(self.bert)
        self.bert.add_adapter("my_lora_adapter", config="lora")
        self.bert.train_adapter("my_lora_adapter")

        if use_half_precision:
            import apex
            self.bert = self.bert.half()

        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.max_length = max_length

    def forward(self, query: List[torch.Tensor], document_batch: List[torch.Tensor]):
        target_device = next(self.parameters()).device
        cls_token = torch.tensor([self.cls_token_id], device=target_device)
        sep_token = torch.tensor([self.sep_token_id], device=target_device)
        input_tensors, position_ids = [], []

        for q, d in zip(query, document_batch):
            d = d[:max(0, self.max_length - len(q) - 2)]
            combined_tensor = torch.cat([cls_token, q, sep_token, d], dim=0)
            input_tensors.append(combined_tensor)
            position_ids.append(torch.arange(combined_tensor.size(0), device=target_device))

        bert_input = PaddedSequence.autopad(input_tensors, batch_first=True, padding_value=self.pad_token_id, device=target_device)
        outputs = self.bert(bert_input.data, attention_mask=bert_input.mask(on=1.0, off=0.0, dtype=torch.float, device=target_device))
        return outputs.logits
