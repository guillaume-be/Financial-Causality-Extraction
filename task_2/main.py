import logging
import os
from pathlib import Path

import torch
from transformers import DistilBertTokenizer

from task_2.library.evaluation import evaluate
from task_2.library.model import DistilBertForCauseEffect
from task_2.library.preprocessing import load_and_cache_examples
from task_2.library.training import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_TYPE = "distilbert"
MODEL_NAME_OR_PATH = "distilbert-base-uncased-distilled-squad"

DO_TRAIN = False
DO_EVAL = True
# Preprocessing
DO_LOWER_CASE = True  # Set to False for case-sensitive models
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
OVERWRITE_CACHE = False
# Training
PER_GPU_BATCH_SIZE = 12
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_STEPS = 20
LEARNING_RATE = 3e-5
NUM_TRAIN_EPOCHS = 3
# Evaluation
PER_GPU_EVAL_BATCH_SIZE = 8
N_BEST_SIZE = 20
MAX_ANSWER_LENGTH = 300

TRAIN_FILE = Path("E:/Coding/finNLP/task_2/data/fnp2020-train.csv")
PREDICT_FILE = Path("E:/Coding/finNLP/task_2/data/fnp2020-dev.csv")
OUTPUT_DIR = 'E:/Coding/finNLP/task_2/output/'

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training
    if DO_TRAIN:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME_OR_PATH,
                                                        do_lower_case=DO_LOWER_CASE,
                                                        cache_dir=OUTPUT_DIR)
        # ToDo: test loading the weights from the QA layer as well
        model = DistilBertForCauseEffect.from_pretrained(MODEL_NAME_OR_PATH).to(device)

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
                                     save_steps=500,
                                     evaluate_during_training=False,
                                     max_seq_length=MAX_SEQ_LENGTH,
                                     doc_stride=DOC_STRIDE,
                                     eval_batch_size=PER_GPU_EVAL_BATCH_SIZE,
                                     n_best_size=N_BEST_SIZE,
                                     max_answer_length=MAX_ANSWER_LENGTH,
                                     do_lower_case=DO_LOWER_CASE,
                                     learning_rate=LEARNING_RATE)

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info("Saving final model to %s", OUTPUT_DIR)

    if DO_EVAL:
        tokenizer = DistilBertTokenizer.from_pretrained(OUTPUT_DIR,
                                                        do_lower_case=DO_LOWER_CASE)
        model = DistilBertForCauseEffect.from_pretrained(OUTPUT_DIR).to(device)

        result = evaluate(model, tokenizer, device, PREDICT_FILE, MODEL_TYPE, MODEL_NAME_OR_PATH,
                          MAX_SEQ_LENGTH, DOC_STRIDE, PER_GPU_EVAL_BATCH_SIZE, OUTPUT_DIR,
                          N_BEST_SIZE, MAX_ANSWER_LENGTH, DO_LOWER_CASE)
        pass
