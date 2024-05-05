# much code in this module is borrowed from the Eraser Benchmark (DeYoung et al., 2019)
from os.path import join, dirname, abspath
import sys
import json
import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoTokenizer, AutoModelForMaskedLM, PretrainedConfig
import adapters
from evidence_inference.models.utils import PaddedSequence

# Adjusting system path to include the module directory
sys.path.insert(0, abspath(join(dirname(__file__), '..', '..')))

def initialize_models(params: dict):
    max_length = params['max_length']
    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
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
        config_options.update({
            'config': PretrainedConfig.from_dict(config_data, num_labels=len(evidence_classes)),
            'bert_dir': None
        })
    else:
        config_options['bert_dir'] = params['bert_dir']

    evidence_identifier = BertClassifier(num_labels=2, **config_options)
    evidence_classifier = BertClassifier(num_labels=len(evidence_classes), **config_options)

    word_interner = tokenizer.get_vocab()
    de_interner = {v: k for k, v in word_interner.items()}
    return evidence_identifier, evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer

class BertClassifier(nn.Module):
    def __init__(self, bert_dir: Optional[str], num_labels: int, pad_token_id: int, cls_token_id: int, sep_token_id: int, max_length: int = 512, use_half_precision: bool = False, config: Optional[PretrainedConfig] = None):
        super().__init__()
        self.bert = (AutoModelForMaskedLM(config) if config else AutoModelForMaskedLM.from_pretrained(bert_dir, num_labels=num_labels))
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

    def forward(self, query: torch.Tensor, document_batch: torch.Tensor):
        assert len(query) == len(document_batch)
        target_device = next(self.parameters()).device
        cls_token = torch.tensor([self.cls_token_id])
        sep_token = torch.tensor([self.sep_token_id])
        input_tensors, position_ids = [], []

        for q, d in zip(query, document_batch):
            truncated_doc = d[:max(0, self.max_length - len(q) - 2)]
            combined_tensor = torch.cat([cls_token, q, sep_token, truncated_doc.to(dtype=q.dtype)])
            input_tensors.append(combined_tensor)
            position_ids.append(torch.arange(combined_tensor.size(0)))

        bert_input = PaddedSequence.autopad(input_tensors, batch_first=True, padding_value=self.pad_token_id, device=target_device)
        outputs = self.bert(bert_input.data, attention_mask=bert_input.mask(on=1.0, off=0.0, dtype=torch.float, device=target_device))
        return outputs.logits

