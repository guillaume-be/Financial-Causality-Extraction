# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForQuestionAnswering

# Set-up training arguments
from SQuAD.library.preprocessing import load_and_cache_examples
from SQuAD.library.training import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_TYPE = "distilbert"
MODEL_NAME_OR_PATH = "distilbert-base-uncased"
DO_TRAIN = True
DO_EVAL = True

TRAIN_FILE = 'E:/Coding/finNLP/SQuAD/data/train-v1.1.json'
PREDICT_FILE = 'E:/Coding/finNLP/SQuAD/data/dev-v1.1.json'
# Preprocessing
DO_LOWER_CASE = True
MAX_SEQ_LENGTH = 384
MAX_QUERY_LENGTH = 64
DOC_STRIDE = 128
# Training
PER_GPU_BATCH_SIZE = 12
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_STEPS = 20
LEARNING_RATE = 3e-5
NUM_TRAIN_EPOCHS = 2
# Evaluation
PER_GPU_EVAL_BATCH_SIZE = 8
N_BEST_SIZE = 20
MAX_ANSWER_LENGTH = 30
VERBOSE_LOGGING = False

OUTPUT_DIR = 'E:/Coding/finNLP/SQuAD/output/'

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = DistilBertConfig.from_pretrained(MODEL_NAME_OR_PATH)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME_OR_PATH,
                                                    do_lower_case=DO_LOWER_CASE,
                                                    cache_dir=OUTPUT_DIR)
    model = DistilBertForQuestionAnswering.from_pretrained(MODEL_NAME_OR_PATH,
                                                           config=config,
                                                           cache_dir=OUTPUT_DIR).to(device)

    # Pre-processing
    input_dir = '/'.join(TRAIN_FILE.split('/')[:-1])

    train_dataset = load_and_cache_examples(input_dir, MODEL_NAME_OR_PATH, tokenizer,
                                            MAX_SEQ_LENGTH, DOC_STRIDE, MAX_QUERY_LENGTH,
                                            output_examples=False, evaluate=False)
    val_dataset, val_examples, val_features = load_and_cache_examples(input_dir, MODEL_NAME_OR_PATH, tokenizer,
                                                                      MAX_SEQ_LENGTH, DOC_STRIDE, MAX_QUERY_LENGTH,
                                                                      output_examples=True, evaluate=True)

    # Training
    global_step, tr_loss = train(train_dataset=train_dataset,
                                 model=model,
                                 tokenizer=tokenizer,
                                 train_batch_size=PER_GPU_BATCH_SIZE,
                                 model_type=MODEL_TYPE,
                                 model_name_or_path=MODEL_NAME_OR_PATH,
                                 input_dir=input_dir,
                                 output_dir=OUTPUT_DIR,
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
                                 max_query_length=MAX_QUERY_LENGTH,
                                 eval_batch_size=PER_GPU_EVAL_BATCH_SIZE,
                                 n_best_size=N_BEST_SIZE,
                                 max_answer_length=MAX_ANSWER_LENGTH,
                                 do_lower_case=DO_LOWER_CASE,
                                 learning_rate=LEARNING_RATE)
