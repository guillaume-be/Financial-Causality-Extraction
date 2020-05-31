# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2020, Guillaume Becquin
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
import multiprocessing
from functools import partial
from pathlib import Path
from typing import Union, List

import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from .data import _is_punctuation, FinCausalClassificationExample, FinCausalClassificationFeature

logger = logging.getLogger(__name__)


def load_and_cache_classification_examples(file_path: Path,
                                           tokenizer,
                                           max_seq_length: int,
                                           output_examples: bool = True,
                                           evaluate=False):
    processor = FinCausalClassificationProcessor()
    examples = processor.get_examples(file_path)

    features, dataset = fincausal_convert_examples_to_classification_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        is_training=not evaluate,
        return_dataset="pt",
        threads=1,
    )

    if output_examples:
        return dataset, examples, features
    return dataset


class FinCausalClassificationProcessor:

    def get_examples(self, file_path: Path):
        input_data = pd.read_csv(file_path, index_col=0, delimiter=';', header=0, skipinitialspace=True)
        input_data.columns = [col_name.strip() for col_name in input_data.columns]
        return self._create_examples(input_data)

    @staticmethod
    def _create_examples(input_data):
        examples = []
        for entry in tqdm(input_data.itertuples()):
            context_text = entry.Text
            example_id = entry.Index
            cause_text = entry.Cause
            effect_text = entry.Effect

            example_cause = FinCausalClassificationExample(example_id, context_text, cause_text, 0)
            example_effect = FinCausalClassificationExample(example_id, context_text, effect_text, 1)
            examples.append(example_cause)
            examples.append(example_effect)

        return examples


def fincausal_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def fincausal_convert_example_to_classification_features(example: FinCausalClassificationExample, max_seq_length: int):
    encoded_dict = tokenizer.encode_plus(text=example.context_text,
                                         text_pair=example.clause_text,
                                         max_length=max_seq_length,
                                         return_overflowing_tokens=False,
                                         pad_to_max_length=True,
                                         stride=0,
                                         truncation_strategy="only_first",
                                         return_token_type_ids=True,
                                         )

    return FinCausalClassificationFeature(
        encoded_dict["input_ids"],
        encoded_dict["attention_mask"],
        encoded_dict["token_type_ids"],
        example.clause_label
    )


def fincausal_convert_examples_to_classification_features(examples: List[FinCausalClassificationExample],
                                                          tokenizer,
                                                          max_seq_length: int,
                                                          is_training: bool,
                                                          return_dataset: Union[bool, str] = False,
                                                          threads: int = 1
                                                          ):
    with multiprocessing.Pool(threads,
                              initializer=fincausal_convert_example_to_features_init,
                              initargs=(tokenizer,)) as p:
        annotate_ = partial(fincausal_convert_example_to_classification_features, max_seq_length=max_seq_length)
        features: List[FinCausalClassificationFeature] = list(tqdm(p.imap(annotate_, examples, chunksize=32),
                                                                   total=len(examples),
                                                                   desc="convert squad examples to features",
                                                                   position=0,
                                                                   leave=True
                                                                   )
                                                              )

    if return_dataset == "pt":
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

        if not is_training:
            dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids)
        else:
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids,
                                    all_attention_masks,
                                    all_token_type_ids,
                                    all_labels,
                                    )

        return features, dataset
    return features


def _run_split_on_punc(text):
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return " ".join(["".join(x) for x in output])
