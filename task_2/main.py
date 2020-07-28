import json
import logging
import os
from pathlib import Path
from enum import Enum
import torch
from transformers import BertTokenizer, DistilBertTokenizer, XLNetTokenizer, AutoTokenizer, RobertaTokenizer, \
    AlbertTokenizer
from transformers import AdamW, get_cosine_schedule_with_warmup

from library.evaluation import evaluate, predict
from library.models.albert import AlbertForCauseEffect
from library.models.bert import BertForCauseEffect
from library.models.distilbert import DistilBertForCauseEffect
from library.models.roberta import RoBERTaForCauseEffect, RoBERTaForCauseEffectClassification
from library.models.xlnet import XLNetForCauseEffect
from library.preprocessing import load_and_cache_examples
from library.training import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


model_config = ModelConfigurations.DistilBert
RUN_NAME = 'FULL_90pc_TRAIN_EVAL/checkpoint-2601'

# model_config = ModelConfigurations.DistilBertSquad
# RUN_NAME = 'FULL_TRAIN_EVAL'

DO_TRAIN = False
DO_EVAL = False
DO_TEST = True
# Preprocessing
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
OVERWRITE_CACHE = True
# Training
PER_GPU_BATCH_SIZE = 4  # 4 for BERT-based models, 12 for DistilBERT
GRADIENT_ACCUMULATION_STEPS = 3  # 3 for BERT-base models, 1 for DistilBERT
WARMUP_STEPS = 50
LEARNING_RATE = 3e-5
DIFFERENTIAL_LR_RATIO = 1.0
NUM_TRAIN_EPOCHS = 5
SAVE_MODEL = True
WEIGHT_DECAY = 0.0
# OPTIMIZER_CLASS = optim.RAdam
OPTIMIZER_CLASS = AdamW
# SCHEDULER_FUNCTION = partial(get_cosine_with_hard_restarts_schedule_with_warmup, num_cycles=2)
# SCHEDULER_FUNCTION = get_linear_schedule_with_warmup
SCHEDULER_FUNCTION = get_cosine_schedule_with_warmup
# Evaluation
PER_GPU_EVAL_BATCH_SIZE = 8
N_BEST_SIZE = 5
MAX_ANSWER_LENGTH = 300
SENTENCE_BOUNDARY_HEURISTIC = True
FULL_SENTENCE_HEURISTIC = True
SHARED_SENTENCE_HEURISTIC = False
POST_CLASSIFICATION = False

(MODEL_TYPE, MODEL_NAME_OR_PATH, DO_LOWER_CASE) = model_config.value
PRACTICE_FILE = Path("E:/Coding/finNLP/task_2/data/fnp2020-fincausal2-task2.csv")
TRIAL_FILE = Path("E:/Coding/finNLP/task_2/data/fnp2020-fincausal-task2.csv")
# TRAIN_SPLIT_FILE = Path("E:/Coding/finNLP/task_2/data/fnp2020-train.csv")
# EVAL_SPLIT_FILE = Path("E:/Coding/finNLP/task_2/data/fnp2020-eval.csv")
TRAIN_SPLIT_FILE = Path("E:/Coding/finNLP/task_2/data/fnp2020-train-90pc.csv")
EVAL_SPLIT_FILE = Path("E:/Coding/finNLP/task_2/data/fnp2020-eval-90pc.csv")
# TRAIN_SPLIT_FILE = Path("E:/Coding/finNLP/task_2/data/fnp2020-train-90pc.csv")
# EVAL_SPLIT_FILE = Path("E:/Coding/finNLP/task_2/data/fnp2020-eval-90pc_subset.csv")
TEST_FILE = Path("E:/Coding/finNLP/task_2/data/task2.csv")

if RUN_NAME is not None:
    OUTPUT_DIR = str(Path('E:/Coding/finNLP/task_2/output') / (MODEL_NAME_OR_PATH + '_' + RUN_NAME))
else:
    OUTPUT_DIR = str(Path('E:/Coding/finNLP/task_2/output') / MODEL_NAME_OR_PATH)

TRAIN_FILE = TRAIN_SPLIT_FILE
EVAL_FILE = EVAL_SPLIT_FILE

model_tokenizer_mapping = {
    'distilbert': (DistilBertForCauseEffect, DistilBertTokenizer),
    'bert': (BertForCauseEffect, BertTokenizer),
    'roberta': (RoBERTaForCauseEffect, RobertaTokenizer),
    'xlnet': (XLNetForCauseEffect, XLNetTokenizer),
    'albert': (AlbertForCauseEffect, AlbertTokenizer),
}

model_class, tokenizer_class = model_tokenizer_mapping[MODEL_TYPE]

log_file = {'MODEL_TYPE': MODEL_TYPE,
            'MODEL_CLASS': model_class.__name__,
            'TOKENIZER_CLASS': tokenizer_class.__name__,
            'MODEL_NAME_OR_PATH': MODEL_NAME_OR_PATH,
            'DO_TRAIN': DO_TRAIN,
            'DO_EVAL': DO_EVAL,
            'DO_LOWER_CASE': DO_LOWER_CASE,
            'MAX_SEQ_LENGTH': MAX_SEQ_LENGTH,
            'DOC_STRIDE': DOC_STRIDE,
            'OVERWRITE_CACHE': OVERWRITE_CACHE,
            'PER_GPU_BATCH_SIZE': PER_GPU_BATCH_SIZE,
            'GRADIENT_ACCUMULATION_STEPS': GRADIENT_ACCUMULATION_STEPS,
            'WARMUP_STEPS': WARMUP_STEPS,
            'LEARNING_RATE': LEARNING_RATE,
            'NUM_TRAIN_EPOCHS': NUM_TRAIN_EPOCHS,
            'PER_GPU_EVAL_BATCH_SIZE': PER_GPU_EVAL_BATCH_SIZE,
            'N_BEST_SIZE': N_BEST_SIZE,
            'MAX_ANSWER_LENGTH': MAX_ANSWER_LENGTH,
            'SENTENCE_BOUNDARY_HEURISTIC': SENTENCE_BOUNDARY_HEURISTIC,
            'FULL_SENTENCE_HEURISTIC': FULL_SENTENCE_HEURISTIC,
            'SHARED_SENTENCE_HEURISTIC': SHARED_SENTENCE_HEURISTIC,
            'OPTIMIZER': str(OPTIMIZER_CLASS),
            'WEIGHT_DECAY': WEIGHT_DECAY,
            'SCHEDULER_FUNCTION': str(SCHEDULER_FUNCTION)
            }

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training
    if DO_TRAIN:

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH,
                                                  do_lower_case=DO_LOWER_CASE,
                                                  cache_dir=OUTPUT_DIR)
        model = model_class.from_pretrained(MODEL_NAME_OR_PATH).to(device)

        train_dataset = load_and_cache_examples(TRAIN_FILE, MODEL_NAME_OR_PATH, tokenizer,
                                                MAX_SEQ_LENGTH, DOC_STRIDE,
                                                output_examples=False,
                                                overwrite_cache=OVERWRITE_CACHE)

        global_step, tr_loss = train(train_dataset=train_dataset,
                                     model=model,
                                     tokenizer=tokenizer,
                                     train_batch_size=PER_GPU_BATCH_SIZE,
                                     model_type=MODEL_TYPE,
                                     model_name_or_path=MODEL_NAME_OR_PATH,
                                     output_dir=OUTPUT_DIR,
                                     predict_file=EVAL_FILE,
                                     device=device,
                                     max_steps=None,
                                     gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                                     num_train_epochs=NUM_TRAIN_EPOCHS,
                                     warmup_steps=WARMUP_STEPS,
                                     logging_steps=500,
                                     save_steps=0,
                                     evaluate_during_training=True,
                                     max_seq_length=MAX_SEQ_LENGTH,
                                     doc_stride=DOC_STRIDE,
                                     eval_batch_size=PER_GPU_EVAL_BATCH_SIZE,
                                     n_best_size=N_BEST_SIZE,
                                     max_answer_length=MAX_ANSWER_LENGTH,
                                     sentence_boundary_heuristic=SENTENCE_BOUNDARY_HEURISTIC,
                                     full_sentence_heuristic=FULL_SENTENCE_HEURISTIC,
                                     shared_sentence_heuristic=SHARED_SENTENCE_HEURISTIC,
                                     learning_rate=LEARNING_RATE,
                                     differential_lr_ratio=DIFFERENTIAL_LR_RATIO,
                                     log_file=log_file,
                                     overwrite_cache=OVERWRITE_CACHE,
                                     optimizer_class=OPTIMIZER_CLASS,
                                     weight_decay=WEIGHT_DECAY,
                                     scheduler_function=SCHEDULER_FUNCTION,
                                     )

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        if SAVE_MODEL:
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            logger.info("Saving final model to %s", OUTPUT_DIR)
        logger.info("Saving log file to %s", OUTPUT_DIR)
        with open(os.path.join(OUTPUT_DIR, "logs.json"), 'w') as f:
            json.dump(log_file, f, indent=4)

    if DO_EVAL:
        tokenizer = tokenizer_class.from_pretrained(OUTPUT_DIR, do_lower_case=DO_LOWER_CASE)
        model = model_class.from_pretrained(OUTPUT_DIR).to(device)

        if POST_CLASSIFICATION:
            classifier_model = RoBERTaForCauseEffectClassification.from_pretrained(
                'E:/Coding/finNLP/task_2/output/deepset/roberta-base-squad2_TRAIN_PRACTICE_EVAL_TRIAL_TRAIN_PRACTICE_EVAL_TRIAL_CLASSIFICATION')
            classifier_tokenizer = RobertaTokenizer.from_pretrained(
                'E:/Coding/finNLP/task_2/output/deepset/roberta-base-squad2_TRAIN_PRACTICE_EVAL_TRIAL_TRAIN_PRACTICE_EVAL_TRIAL_CLASSIFICATION')
        else:
            classifier_model = None
            classifier_tokenizer = None

        result = evaluate(model=model,
                          tokenizer=tokenizer,
                          device=device,
                          file_path=EVAL_FILE,
                          model_type=MODEL_TYPE,
                          model_name_or_path=MODEL_NAME_OR_PATH,
                          max_seq_length=MAX_SEQ_LENGTH,
                          doc_stride=DOC_STRIDE,
                          eval_batch_size=PER_GPU_EVAL_BATCH_SIZE,
                          output_dir=OUTPUT_DIR,
                          n_best_size=N_BEST_SIZE,
                          max_answer_length=MAX_ANSWER_LENGTH,
                          sentence_boundary_heuristic=SENTENCE_BOUNDARY_HEURISTIC,
                          full_sentence_heuristic=FULL_SENTENCE_HEURISTIC,
                          shared_sentence_heuristic=SHARED_SENTENCE_HEURISTIC,
                          overwrite_cache=OVERWRITE_CACHE,
                          classifier_model=classifier_model,
                          classifier_tokenizer=classifier_tokenizer
                          )

        print("done")

    if DO_TEST:
        tokenizer = tokenizer_class.from_pretrained(OUTPUT_DIR, do_lower_case=DO_LOWER_CASE)
        model = model_class.from_pretrained(OUTPUT_DIR).to(device)

        if POST_CLASSIFICATION:
            classifier_model = RoBERTaForCauseEffectClassification.from_pretrained(
                'E:/Coding/finNLP/task_2/output/deepset/roberta-base-squad2_TRAIN_PRACTICE_EVAL_TRIAL_TRAIN_PRACTICE_EVAL_TRIAL_CLASSIFICATION')
            classifier_tokenizer = RobertaTokenizer.from_pretrained(
                'E:/Coding/finNLP/task_2/output/deepset/roberta-base-squad2_TRAIN_PRACTICE_EVAL_TRIAL_TRAIN_PRACTICE_EVAL_TRIAL_CLASSIFICATION')
        else:
            classifier_model = None
            classifier_tokenizer = None

        result = predict(model=model,
                         tokenizer=tokenizer,
                         device=device,
                         file_path=TEST_FILE,
                         model_type=MODEL_TYPE,
                         model_name_or_path=MODEL_NAME_OR_PATH,
                         max_seq_length=MAX_SEQ_LENGTH,
                         doc_stride=DOC_STRIDE,
                         eval_batch_size=PER_GPU_EVAL_BATCH_SIZE,
                         output_dir=OUTPUT_DIR,
                         n_best_size=N_BEST_SIZE,
                         max_answer_length=MAX_ANSWER_LENGTH,
                         sentence_boundary_heuristic=SENTENCE_BOUNDARY_HEURISTIC,
                         full_sentence_heuristic=FULL_SENTENCE_HEURISTIC,
                         shared_sentence_heuristic=SHARED_SENTENCE_HEURISTIC,
                         overwrite_cache=OVERWRITE_CACHE,
                         )

        print("done")
