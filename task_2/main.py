import logging
from pathlib import Path

import torch
from transformers import DistilBertTokenizer

from task_2.data import FinCausalFeatures
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
OVERWRITE_CACHE = False

TRAIN_FILE = Path("E:/Coding/finNLP/task_2/data/fnp2020-train.csv")
PREDICT_FILE = Path("E:/Coding/finNLP/task_2/data/fnp2020-dev.csv")
OUTPUT_DIR = 'E:/Coding/finNLP/task_2/output/'

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME_OR_PATH,
                                                    do_lower_case=DO_LOWER_CASE,
                                                    cache_dir=OUTPUT_DIR)

    train_dataset, train_examples, train_features = load_and_cache_examples(TRAIN_FILE, MODEL_NAME_OR_PATH, tokenizer,
                                                                            MAX_SEQ_LENGTH, DOC_STRIDE,
                                                                            output_examples=True,
                                                                            overwrite_cache=OVERWRITE_CACHE)
    # val_dataset, val_examples, val_features = load_and_cache_examples(PREDICT_FILE, MODEL_NAME_OR_PATH, tokenizer,
    #                                                                   MAX_SEQ_LENGTH, DOC_STRIDE,
    #                                                                   output_examples=True, evaluate=True,
    #                                                                   overwrite_cache=OVERWRITE_CACHE)

    #     Validation of pre-processing
    rebuilt_features = []
    for feature, tensors in zip(train_features, train_dataset):
        assert (isinstance(feature, FinCausalFeatures))
        token_ids = tensors[0]
        example_id = feature.example_orig_index
        linked_example = [example for example in train_examples if example.example_id == example_id][0]
        rebuilt_text = tokenizer.decode(token_ids, False, False)
        rebuilt_cause = tokenizer.decode(token_ids[feature.cause_start_position:feature.cause_end_position + 1], False,
                                         False)
        rebuilt_effect = tokenizer.decode(token_ids[feature.effect_start_position:feature.effect_end_position + 1],
                                          False,
                                          False)
        pass
