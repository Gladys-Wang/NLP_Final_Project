# much code in this module is borrowed from the Eraser Benchmark (DeYoung et al., 2019)
from os.path import join, dirname, abspath
import sys
import json
import torch
import torch.nn as nn
from typing import Optional
from transformers import RobertaTokenizer, RobertaForSequenceClassification, PretrainedConfig
import adapters
from evidence_inference.models.utils import PaddedSequence

# Adjusting system path to include the module directory
sys.path.insert(0, abspath(join(dirname(__file__), '..', '..')))

def initialize_models(params: dict, unk_token='<unk>'):
    max_length = params['max_length']
    tokenizer = RobertaTokenizer.from_pretrained(params['bert_vocab'])
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    evidence_classes = {y: x for x, y in params['evidence_classifier']['classes'].items()}

    if params.get('random_init', False):
        with open(params['bert_config'], 'r') as file:
            config_json = json.load(file)
            id_config = PretrainedConfig.from_dict(config_json, num_labels=2)
            cls_config = PretrainedConfig.from_dict(config_json, num_labels=len(evidence_classes))

        bert_cls_args = {
            'pad_token_id': pad_token_id,
            'cls_token_id': cls_token_id,
            'sep_token_id': sep_token_id,
            'max_length': max_length,
            'use_half_precision': params['evidence_identifier'].get('use_half_precision', False),
            'config': id_config
        }
        evidence_identifier = BertClassifier(**bert_cls_args)
        bert_cls_args['config'] = cls_config
        evidence_classifier = BertClassifier(**bert_cls_args)
    else:
        bert_cls_args = {
            'bert_dir': params['bert_dir'],
            'pad_token_id': pad_token_id,
            'cls_token_id': cls_token_id,
            'sep_token_id': sep_token_id,
            'num_labels': len(evidence_classes),
            'max_length': max_length,
            'use_half_precision': params['evidence_classifier'].get('use_half_precision', False)
        }
        evidence_identifier = BertClassifier(**bert_cls_args)
        evidence_classifier = BertClassifier(**bert_cls_args)

    word_interner = tokenizer.get_vocab()
    de_interner = {value: key for key, value in word_interner.items()}
    return evidence_identifier, evidence_classifier, word_interner, de_interner, evidence_classes, tokenizer

class BertClassifier(nn.Module):
    def __init__(self, bert_dir: Optional[str], pad_token_id: int, cls_token_id: int, sep_token_id: int, num_labels: int, max_length: int = 512, use_half_precision: bool = False, config: Optional[PretrainedConfig] = None):
        super().__init__()
        if bert_dir is None:
            assert config, "Config must be provided for model initialization"
            bert = RobertaForSequenceClassification(config)
        else:
            bert = RobertaForSequenceClassification.from_pretrained(bert_dir, num_labels=num_labels)

        adapters.init(bert)
        bert.add_adapter("my_lora_adapter", config="lora")
        bert.train_adapter("my_lora_adapter")

        if use_half_precision:
            bert = bert.half()

        self.bert = bert
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.max_length = max_length

    def forward(self,
                query: List[torch.tensor],
                document_batch: List[torch.tensor]):
        assert len(query) == len(document_batch)
        # note about device management:
        # since distributed training is enabled, the inputs to this module can be on *any* device (preferably cpu, since we wrap and unwrap the module)
        # we want to keep these params on the input device (assuming CPU) for as long as possible for cheap memory access
        target_device = next(self.parameters()).device
        cls_token = torch.tensor([self.cls_token_id])#.to(device=document_batch[0].device)
        sep_token = torch.tensor([self.sep_token_id])#.to(device=document_batch[0].device)
        input_tensors = []
        position_ids = []
        for q, d in zip(query, document_batch):
            if len(q) + len(d) + 2 > self.max_length:
                d = d[:(self.max_length - len(q) - 2)]
            input_tensors.append(torch.cat([cls_token, q, sep_token, d.to(dtype=q.dtype)]))
            position_ids.append(torch.arange(0, input_tensors[-1].size().numel()))
            #position_ids.append(torch.tensor(list(range(0, len(q) + 1)) + list(range(0, len(d) + 1))))
        bert_input = PaddedSequence.autopad(input_tensors, batch_first=True, padding_value=self.pad_token_id, device=target_device)
        positions = PaddedSequence.autopad(position_ids, batch_first=True, padding_value=0, device=target_device)
        outputs = self.bert(bert_input.data, attention_mask=bert_input.mask(on=1.0, off=0.0, dtype=torch.float, device=target_device), position_ids=positions.data)
        classes = outputs.logits
        assert torch.all(classes == classes) # for nans
        return classes
