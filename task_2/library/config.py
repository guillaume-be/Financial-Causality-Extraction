from enum import Enum
from typing import Callable

import torch
from transformers import AdamW, get_cosine_schedule_with_warmup, AlbertTokenizer, XLNetTokenizer, RobertaTokenizer, \
    BertTokenizer, DistilBertTokenizer

from library.models.albert import AlbertForCauseEffect
from library.models.bert import BertForCauseEffect
from library.models.distilbert import DistilBertForCauseEffect
from library.models.roberta import RoBERTaForCauseEffect
from library.models.xlnet import XLNetForCauseEffect


class ModelConfigurations(Enum):
    BertBase = ('bert', 'bert-base-cased', False)
    BertLarge = ('bert', 'bert-large-cased', False)
    BertSquad = ('bert', 'deepset/bert-base-cased-squad2', False)
    BertSquad2 = ('bert', 'deepset/bert-large-uncased-whole-word-masking-squad2', True)
    DistilBert = ('distilbert', 'distilbert-base-uncased', True)
    DistilBertSquad = ('distilbert', 'distilbert-base-uncased-distilled-squad', True)
    RoBERTaSquad = ('roberta', 'deepset/roberta-base-squad2', False)
    RoBERTaSquadLarge = ('roberta', 'ahotrod/roberta_large_squad2', False)
    RoBERTa = ('roberta', 'roberta-base', False)
    RoBERTaLarge = ('roberta', 'roberta-large', False)
    XLNetBase = ('xlnet', 'xlnet-base-cased', False)
    AlbertSquad = ('albert', 'twmkn9/albert-base-v2-squad2', True)


model_tokenizer_mapping = {
    'distilbert': (DistilBertForCauseEffect, DistilBertTokenizer),
    'bert': (BertForCauseEffect, BertTokenizer),
    'roberta': (RoBERTaForCauseEffect, RobertaTokenizer),
    'xlnet': (XLNetForCauseEffect, XLNetTokenizer),
    'albert': (AlbertForCauseEffect, AlbertTokenizer),
}


class RunConfig:
    def __init__(self,
                 do_train: bool = False,
                 do_eval: bool = True,
                 do_test: bool = False,
                 max_seq_length: int = 384,
                 doc_stride: int = 128,
                 train_batch_size: int = 4,
                 gradient_accumulation_steps: int = 3,
                 warmup_steps: int = 50,
                 learning_rate: float = 3e-5,
                 differential_lr_ratio: float = 1.0,
                 max_grad_norm: float = 1.0,
                 adam_epsilon: float = 1e-8,
                 num_train_epochs: int = 5,
                 save_model: bool = True,
                 weight_decay: float = 0.0,
                 optimizer_class: torch.optim.Optimizer = AdamW,
                 scheduler_function: Callable = get_cosine_schedule_with_warmup,
                 evaluate_during_training: bool = True,
                 eval_batch_size: int = 8,
                 n_best_size: int = 5,
                 max_answer_length: int = 300,
                 sentence_boundary_heuristic: bool = True,
                 full_sentence_heuristic: bool = True,
                 shared_sentence_heuristic: bool = False,
                 top_n_sentences: bool = True):
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_test = do_test
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.differential_lr_ratio = differential_lr_ratio
        self.max_grad_norm = max_grad_norm
        self.adam_epsilon = adam_epsilon
        self.num_train_epochs = num_train_epochs
        self.save_model = save_model
        self.weight_decay = weight_decay
        self.optimizer_class = optimizer_class
        self.scheduler_function = scheduler_function
        self.evaluate_during_training = evaluate_during_training
        self.eval_batch_size = eval_batch_size
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.sentence_boundary_heuristic = sentence_boundary_heuristic
        self.full_sentence_heuristic = full_sentence_heuristic
        self.shared_sentence_heuristic = shared_sentence_heuristic
        self.top_n_sentences = top_n_sentences
