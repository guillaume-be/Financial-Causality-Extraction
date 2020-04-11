import logging
from pathlib import Path

import torch
from transformers import DistilBertTokenizer

from task_2.library.preprocessing import load_and_cache_examples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_TYPE = "distilbert"
MODEL_NAME_OR_PATH = "distilbert-base-uncased"

DO_TRAIN = True
DO_EVAL = True
# Preprocessing
DO_LOWER_CASE = True  # Set to False for case-sensitive models
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128

TRAIN_FILE = Path("E:/Coding/YseopLab/FNP_2020_FinCausal/baseline/task2/fnp2020-train.csv")
PREDICT_FILE = Path("E:/Coding/YseopLab/FNP_2020_FinCausal/baseline/task2/fnp2020-dev.csv")
OUTPUT_DIR = 'E:/Coding/finNLP/task_2/output/'

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME_OR_PATH,
    #                                                 do_lower_case=DO_LOWER_CASE,
    #                                                 cache_dir=OUTPUT_DIR)
    tokenizer = None

    train_dataset = load_and_cache_examples(TRAIN_FILE, MODEL_NAME_OR_PATH, tokenizer,
                                            MAX_SEQ_LENGTH, DOC_STRIDE,
                                            output_examples=False, overwrite_cache=True)
