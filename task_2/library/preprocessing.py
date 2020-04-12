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
import os
from functools import partial
from pathlib import Path
from typing import Union, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers.tokenization_bert import whitespace_tokenize

from task_2.library.data import FinCausalExample, FinCausalFeatures

logger = logging.getLogger(__name__)


def load_and_cache_examples(file_path: Path, model_name_or_path: str,
                            tokenizer,
                            max_seq_length: int, doc_stride: int,
                            output_examples: bool = True,
                            evaluate=False, overwrite_cache: bool = False):
    # Load data features from cache or dataset file
    input_dir = file_path.parents[0]
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, model_name_or_path.split("/"))).pop()
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)
        processor = FinCausalProcessor()
        examples = processor.get_examples(file_path)

        features, dataset = fincausal_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            is_training=not evaluate,
            return_dataset="pt",
            # threads=multiprocessing.cpu_count(),
            threads=1,
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


class FinCausalProcessor:

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
            cause_start_position_character = entry.Cause_Start
            cause_end_position_character = entry.Cause_End
            effect_start_position_character = entry.Effect_Start
            effect_end_position_character = entry.Effect_End

            example = FinCausalExample(
                example_id,
                context_text,
                cause_text,
                effect_text,
                cause_start_position_character,
                cause_end_position_character,
                effect_start_position_character,
                effect_end_position_character
            )
            examples.append(example)
        return examples


def fincausal_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def _improve_answer_span(doc_tokens: List[str],
                         input_start: int,
                         input_end: int,
                         tokenizer,
                         orig_answer_text: str):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def _new_check_is_max_context(doc_spans: List[dict],
                              cur_span_index: int,
                              position: int):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def fincausal_convert_example_to_features(example: FinCausalExample,
                                          max_seq_length: int,
                                          doc_stride: int,
                                          is_training: bool):
    features = []
    if is_training:
        # Get start and end position
        start_cause_position = example.start_cause_position
        end_cause_position = example.end_cause_position
        start_effect_position = example.start_effect_position
        end_effect_position = example.end_effect_position

        # If the cause cannot be found in the text, then skip this example.
        actual_cause_text = " ".join(example.doc_tokens[start_cause_position: (end_cause_position + 1)])
        cleaned_cause_text = " ".join(whitespace_tokenize(example.cause_text))
        if actual_cause_text.find(cleaned_cause_text) == -1:
            logger.warning("Could not find cause: '%s' vs. '%s'", actual_cause_text, cleaned_cause_text)
            return []

        # If the effect cannot be found in the text, then skip this example.
        actual_effect_text = " ".join(example.doc_tokens[start_effect_position: (end_effect_position + 1)])
        cleaned_effect_text = " ".join(whitespace_tokenize(example.effect_text))
        if actual_effect_text.find(cleaned_effect_text) == -1:
            logger.warning("Could not find effect: '%s' vs. '%s'", actual_effect_text, cleaned_effect_text)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training:
        tok_cause_start_position = orig_to_tok_index[example.start_cause_position]
        if example.end_cause_position < len(example.doc_tokens) - 1:
            tok_cause_end_position = orig_to_tok_index[example.end_cause_position + 1] - 1
        else:
            tok_cause_end_position = len(all_doc_tokens) - 1

        (tok_cause_start_position, tok_cause_end_position) = _improve_answer_span(
            all_doc_tokens, tok_cause_start_position, tok_cause_end_position, tokenizer, example.cause_text
        )

        tok_effect_start_position = orig_to_tok_index[example.start_effect_position]
        if example.end_effect_position < len(example.doc_tokens) - 1:
            tok_effect_end_position = orig_to_tok_index[example.end_effect_position + 1] - 1
        else:
            tok_effect_end_position = len(all_doc_tokens) - 1

        (tok_effect_start_position, tok_effect_end_position) = _improve_answer_span(
            all_doc_tokens, tok_effect_start_position, tok_effect_end_position, tokenizer, example.effect_text
        )

    spans = []

    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence
        if "roberta" in str(type(tokenizer)) or "camembert" in str(type(tokenizer))
        else tokenizer.max_len - tokenizer.max_len_single_sentence - 1
    )

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        encoded_dict = tokenizer.encode_plus(span_doc_tokens,
                                             max_length=max_seq_length,
                                             return_overflowing_tokens=True,
                                             pad_to_max_length=True,
                                             stride=max_seq_length - doc_stride - sequence_added_tokens - 1,
                                             truncation_strategy="only_first",
                                             return_token_type_ids=True,
                                             )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - sequence_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                        len(encoded_dict["input_ids"])
                        - 1
                        - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]
        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for index in range(paragraph_len):
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + index]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict:
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            spans[doc_span_index]["token_is_max_context"][j] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = np.ones(len(span["token_type_ids"]))
        p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1
        # Set the CLS index to '0'
        p_mask[cls_index] = 0

        span_is_impossible = False
        cause_start_position = 0
        cause_end_position = 0
        effect_start_position = 0
        effect_end_position = 0
        if is_training:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_cause_start_position >= doc_start
                    and tok_cause_end_position <= doc_end
                    and tok_effect_start_position >= doc_start
                    and tok_effect_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                cause_start_position = cls_index
                cause_end_position = cls_index
                effect_start_position = cls_index
                effect_end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = sequence_added_tokens

                cause_start_position = tok_cause_start_position - doc_start + doc_offset
                cause_end_position = tok_cause_end_position - doc_start + doc_offset
                effect_start_position = tok_effect_start_position - doc_start + doc_offset
                effect_end_position = tok_effect_end_position - doc_start + doc_offset

        features.append(
            FinCausalFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_orig_index=example.example_id,
                example_index=0,
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                cause_start_position=cause_start_position,
                cause_end_position=cause_end_position,
                effect_start_position=effect_start_position,
                effect_end_position=effect_end_position,
                is_impossible=span_is_impossible,
            )
        )
    return features


def fincausal_convert_examples_to_features(
        examples: List[FinCausalExample], tokenizer, max_seq_length: int, doc_stride: int, is_training: bool,
        return_dataset: Union[bool, str] = False, threads: int = 1
):
    with multiprocessing.Pool(threads,
                              initializer=fincausal_convert_example_to_features_init,
                              initargs=(tokenizer,)) as p:
        annotate_ = partial(
            fincausal_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            is_training=is_training,
        )
        features: List[FinCausalFeatures] = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
            )
        )
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id"):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features
    if return_dataset == "pt":

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

        if not is_training:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask
            )
        else:
            all_cause_start_positions = torch.tensor([f.cause_start_position for f in features], dtype=torch.long)
            all_cause_end_positions = torch.tensor([f.cause_end_position for f in features], dtype=torch.long)
            all_effect_start_positions = torch.tensor([f.effect_start_position for f in features], dtype=torch.long)
            all_effect_end_positions = torch.tensor([f.effect_end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_cause_start_positions,
                all_cause_end_positions,
                all_effect_start_positions,
                all_effect_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
            )

        return features, dataset
    return features
