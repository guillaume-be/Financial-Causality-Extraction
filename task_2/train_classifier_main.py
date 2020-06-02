import json
import logging
import os
from pathlib import Path
from enum import Enum
import torch
from transformers import RobertaTokenizer, BertTokenizer

from library.evaluation import evaluate
from library.evaluation_classifier import evaluate_classifier
from library.models.bert import BertForCauseEffectClassification
from library.models.roberta import RoBERTaForCauseEffectClassification
from library.preprocessing import load_and_cache_examples
from library.preprocessing_classifier import load_and_cache_classification_examples
from library.training_classifier import train_classifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfigurations(Enum):
    RoBERTaSquad = ('roberta',
                    'E:/Coding/finNLP/task_2/output/deepset/roberta-base-squad2_TRAIN_PRACTICE_EVAL_TRIAL',
                    False),
    FinBERT = ('finbert',
               'E:/Coding/finNLP/task_2/pretrained/finbert',
               True),
    FinBERTSentiment = ('finbert-sentiment',
                        'E:/Coding/finNLP/task_2/pretrained/finbert_sentiment',
                        True)


model_config = ModelConfigurations.FinBERTSentiment
RUN_NAME = 'TRAIN_PRACTICE_EVAL_TRIAL_CLASSIFICATION'

DO_TRAIN = True
DO_EVAL = False
# Preprocessing
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
OVERWRITE_CACHE = True
# Training
PER_GPU_BATCH_SIZE = 4  # 4 for BERT-based models, 12 for DistilBERT
GRADIENT_ACCUMULATION_STEPS = 3  # 3 for BERT-base models, 1 for DistilBERT
WARMUP_STEPS = 20
LEARNING_RATE = 3e-5
DIFFERENTIAL_LR_RATIO = 1.0
NUM_TRAIN_EPOCHS = 5
SAVE_MODEL = True
# Evaluation
PER_GPU_EVAL_BATCH_SIZE = 8
N_BEST_SIZE = 5
MAX_ANSWER_LENGTH = 300
SENTENCE_BOUNDARY_HEURISTIC = True
FULL_SENTENCE_HEURISTIC = False
SHARED_SENTENCE_HEURISTIC = False

(MODEL_TYPE, MODEL_NAME_OR_PATH, DO_LOWER_CASE) = model_config.value
PRACTICE_FILE = Path("E:/Coding/finNLP/task_2/data/fnp2020-fincausal2-task2.csv")
TRIAL_FILE = Path("E:/Coding/finNLP/task_2/data/fnp2020-fincausal-task2.csv")

if RUN_NAME is not None:
    OUTPUT_DIR = str(Path('E:/Coding/finNLP/task_2/output') / (MODEL_NAME_OR_PATH + '_' + RUN_NAME))
else:
    OUTPUT_DIR = str(Path('E:/Coding/finNLP/task_2/output') / MODEL_NAME_OR_PATH)

TRAIN_FILE = PRACTICE_FILE
PREDICT_FILE = TRIAL_FILE

model_tokenizer_mapping = {
    'roberta': (RoBERTaForCauseEffectClassification, RobertaTokenizer),
    'finbert': (BertForCauseEffectClassification, BertTokenizer),
    'finbert-sentiment': (BertForCauseEffectClassification, BertTokenizer),
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

        train_dataset, examples, features = load_and_cache_classification_examples(TRAIN_FILE,
                                                                                   tokenizer,
                                                                                   MAX_SEQ_LENGTH,
                                                                                   output_examples=True)

        global_step, tr_loss = train_classifier(train_dataset=train_dataset,
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
                                                eval_batch_size=PER_GPU_EVAL_BATCH_SIZE,
                                                learning_rate=LEARNING_RATE,
                                                differential_lr_ratio=DIFFERENTIAL_LR_RATIO,
                                                log_file=log_file)

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

        result = evaluate_classifier(model=model,
                                     tokenizer=tokenizer,
                                     device=device,
                                     file_path=PREDICT_FILE,
                                     model_type=MODEL_TYPE,
                                     max_seq_length=MAX_SEQ_LENGTH,
                                     eval_batch_size=PER_GPU_EVAL_BATCH_SIZE,
                                     output_dir=OUTPUT_DIR)
