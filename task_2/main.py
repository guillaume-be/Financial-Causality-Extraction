import logging
import os
from pathlib import Path
from enum import Enum
import torch
from transformers import BertTokenizer, DistilBertTokenizer, XLNetTokenizer

from task_2.library.evaluation import evaluate
from task_2.library.models.bert import BertForCauseEffect
from task_2.library.models.distilbert import DistilBertForCauseEffect
from task_2.library.models.roberta import RoBERTaForCauseEffect
from task_2.library.models.xlnet import XLNetForCauseEffect
from task_2.library.preprocessing import load_and_cache_examples
from task_2.library.training import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfigurations(Enum):
    BertSquad = ('bert', 'deepset/bert-base-cased-squad2', False)
    DistilBertSquad = ('distilbert', 'distilbert-base-uncased-distilled-squad', True)
    RoBERTaSquad = ('roberta', 'deepset/roberta-base-squad2', False)
    XLNetBase = ('xlnet', 'xlnet-base-cased', False)


DO_TRAIN = False
DO_EVAL = True
# Preprocessing
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
OVERWRITE_CACHE = True
# Training
PER_GPU_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 3
WARMUP_STEPS = 20
LEARNING_RATE = 3e-5
DIFFERENTIAL_LR_RATIO = 1.0
NUM_TRAIN_EPOCHS = 3
SAVE_MODEL = False
# Evaluation
PER_GPU_EVAL_BATCH_SIZE = 8
N_BEST_SIZE = 5
MAX_ANSWER_LENGTH = 300
SENTENCE_BOUNDARY_HEURISTIC = True
FULL_SENTENCE_HEURISTIC = False
SHARED_SENTENCE_HEURISTIC = False

model_config = ModelConfigurations.DistilBertSquad
(MODEL_TYPE, MODEL_NAME_OR_PATH, DO_LOWER_CASE) = model_config.value
TRAIN_FILE = Path("E:/Coding/finNLP/task_2/data/fnp2020-fincausal2-task2.csv")
PREDICT_FILE = Path("E:/Coding/finNLP/task_2/data/fnp2020-fincausal-task2.csv")
OUTPUT_DIR = 'E:/Coding/finNLP/task_2/output/' + MODEL_NAME_OR_PATH

model_tokenizer_mapping = {
    'distilbert': (DistilBertForCauseEffect, DistilBertTokenizer),
    'bert': (BertForCauseEffect, BertTokenizer),
    'roberta': (RoBERTaForCauseEffect, BertTokenizer),
    'xlnet': (XLNetForCauseEffect, XLNetTokenizer),
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
            'SHARED_SENTENCE_HEURISTIC': SHARED_SENTENCE_HEURISTIC
            }

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training
    if DO_TRAIN:

        tokenizer = tokenizer_class.from_pretrained(MODEL_NAME_OR_PATH,
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
                                     predict_file=PREDICT_FILE,
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
                                     overwrite_cache=OVERWRITE_CACHE)

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        if SAVE_MODEL:
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            logger.info("Saving final model to %s", OUTPUT_DIR)

    if DO_EVAL:
        tokenizer = tokenizer_class.from_pretrained(OUTPUT_DIR,
                                                    do_lower_case=DO_LOWER_CASE)
        model = model_class.from_pretrained(OUTPUT_DIR).to(device)

        result = evaluate(model=model,
                          tokenizer=tokenizer,
                          device=device,
                          file_path=PREDICT_FILE,
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
                          overwrite_cache=OVERWRITE_CACHE)
