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
import collections
import csv
import json
import logging
import math
import os
import timeit
from enum import Enum
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import BasicTokenizer

from .fincausal_evaluation.task2_evaluate import encode_causal_tokens, Task2Data
from .data import FinCausalResult, FinCausalFeatures, FinCausalExample
from .preprocessing import load_and_cache_examples
from .fincausal_evaluation.task2_evaluate import evaluate as official_evaluate
from .preprocessing_classifier import load_and_cache_classification_examples

logger = logging.getLogger(__name__)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate_classifier(model, tokenizer, device: torch.device, file_path: Path, model_type: str,
                        max_seq_length: int, eval_batch_size: int, output_dir: str, prefix=""):
    dataset, _, features = load_and_cache_classification_examples(file_path,
                                                                  tokenizer,
                                                                  max_seq_length,
                                                                  output_examples=True,
                                                                  evaluate=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    all_results = []

    for batch in tqdm(eval_dataloader, desc="Evaluating", position=0, leave=True):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            outputs = list(model(**inputs)[0].cpu().softmax(dim=1).numpy())
            all_results.extend(outputs)

    total = 0
    correct = 0
    for predicted, feature in zip(all_results, features):
        if predicted[0] > 0.5:
            predicted_class = 0
        else:
            predicted_class = 1
        if predicted_class == feature.label:
            correct += 1
        total += 1
    result = {'accuracy': (correct / total)}
    print(result)
    return result


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
