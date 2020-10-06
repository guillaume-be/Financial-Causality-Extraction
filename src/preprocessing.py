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
from collections import UserDict
from functools import partial
from pathlib import Path
from typing import Union, List, Tuple

import spacy
from pysbd.utils import PySBDFactory

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, BatchEncoding, PreTrainedTokenizerFast
from transformers.tokenization_bert import whitespace_tokenize

from .config import RunConfig
from .data import FinCausalExample, FinCausalFeatures, _is_punctuation

logger = logging.getLogger(__name__)


def load_examples(file_path: Path,
                  tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                  run_config: RunConfig,
                  output_examples: bool = True,
                  evaluate: bool = False) -> \
        Union[Tuple[TensorDataset, List[FinCausalExample], List[FinCausalFeatures]],
              TensorDataset]:
    processor = FinCausalProcessor()
    examples = processor.get_examples(file_path)

    features, dataset = fincausal_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=run_config.max_seq_length,
        doc_stride=run_config.doc_stride,
        is_training=not evaluate,
        return_dataset="pt",
        threads=1,
    )

    if output_examples:
        return dataset, examples, features
    return dataset


class FinCausalProcessor:

    def get_examples(self, file_path: Path) -> List[FinCausalExample]:
        input_data = pd.read_csv(file_path, index_col=0, delimiter=';', header=0, skipinitialspace=True)
        input_data.columns = [col_name.strip() for col_name in input_data.columns]
        return self._create_examples(input_data)

    @staticmethod
    def _create_examples(input_data: pd.DataFrame) -> List[FinCausalExample]:

        nlp = spacy.blank('en')
        nlp.add_pipe(PySBDFactory(nlp))

        examples = []
        for entry in tqdm(input_data.itertuples()):
            context_text = entry.Text
            example_id = entry.Index
            if len(entry) > 2:
                cause_text = entry.Cause
                effect_text = entry.Effect
                cause_start_position_character = entry.Cause_Start
                cause_end_position_character = entry.Cause_End
                effect_start_position_character = entry.Effect_Start
                effect_end_position_character = entry.Effect_End
            else:
                cause_text = ""
                effect_text = ""
                cause_start_position_character = 0
                cause_end_position_character = 0
                effect_start_position_character = 0
                effect_end_position_character = 0

            doc = nlp(entry.Text)
            sentences = list(doc.sents)
            offset_sentence_2 = np.nan
            offset_sentence_3 = np.nan
            if len(sentences) > 1:
                offset_sentence_2 = sentences[0].end_char
            if len(sentences) > 2:
                offset_sentence_3 = sentences[1].end_char

            example = FinCausalExample(
                example_id,
                context_text,
                cause_text,
                effect_text,
                offset_sentence_2,
                offset_sentence_3,
                cause_start_position_character,
                cause_end_position_character,
                effect_start_position_character,
                effect_end_position_character
            )
            examples.append(example)
        return examples


def fincausal_convert_example_to_features_init(
        tokenizer_for_convert: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]):
    global tokenizer
    tokenizer = tokenizer_for_convert


def _improve_answer_span(doc_tokens: List[str],
                         input_start: int,
                         input_end: int,
                         tokenizer,
                         orig_answer_text: str) -> Tuple[int, int]:
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def _check_is_max_context(doc_spans: Union[List[dict], List[UserDict]],
                          cur_span_index: int,
                          position: int) -> bool:
    """Check if this is the 'max context' doc span for the token."""
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
                                          is_training: bool) -> List[FinCausalFeatures]:
    features = []
    if is_training:
        # Get start and end position
        start_cause_position = example.start_cause_position
        end_cause_position = example.end_cause_position
        start_effect_position = example.start_effect_position
        end_effect_position = example.end_effect_position

        # If the cause cannot be found in the text, then skip this example.
        actual_cause_text = " ".join(example.doc_tokens[start_cause_position: (end_cause_position + 1)])
        cleaned_cause_text = " ".join(whitespace_tokenize(_run_split_on_punc(example.cause_text)))
        if actual_cause_text.find(cleaned_cause_text) == -1:
            logger.warning("Could not find cause: '%s' vs. '%s'", actual_cause_text, cleaned_cause_text)
            return []

        # If the effect cannot be found in the text, then skip this example.
        actual_effect_text = " ".join(example.doc_tokens[start_effect_position: (end_effect_position + 1)])
        cleaned_effect_text = " ".join(whitespace_tokenize(_run_split_on_punc(example.effect_text)))
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
    if example.offset_sentence_2 > 0:
        tok_sentence_2_offset = orig_to_tok_index[example.offset_sentence_2 + 1] - 1
    else:
        tok_sentence_2_offset = None
    if example.offset_sentence_3 > 0:
        tok_sentence_3_offset = orig_to_tok_index[example.offset_sentence_3 + 1] - 1
    else:
        tok_sentence_3_offset = None

    spans: List[BatchEncoding] = []

    sequence_added_tokens = tokenizer.max_len - tokenizer.max_len_single_sentence

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        encoded_dict: BatchEncoding = tokenizer.encode_plus(span_doc_tokens,
                                                            max_length=max_seq_length,
                                                            return_overflowing_tokens=True,
                                                            pad_to_max_length=True,
                                                            stride=max_seq_length - doc_stride - sequence_added_tokens - 1,
                                                            truncation_strategy="only_first",
                                                            truncation=True,
                                                            return_token_type_ids=True,
                                                            )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - sequence_added_tokens,
        )
        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict.data["input_ids"][
                                 : encoded_dict.data["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                        len(encoded_dict.data["input_ids"])
                        - 1
                        - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]
        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = sequence_added_tokens + i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if len(encoded_dict.get("overflowing_tokens", [])) == 0:
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index].data["paragraph_len"]):
            is_max_context = _check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            spans[doc_span_index].data["token_is_max_context"][j] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span.data["input_ids"].index(tokenizer.cls_token_id)

        p_mask = np.ones(len(span.data["token_type_ids"]))
        p_mask[np.where(np.array(span.data["input_ids"]) == tokenizer.sep_token_id)[0]] = 1
        # Set the CLS index to '0'
        p_mask[cls_index] = 0

        span_is_impossible = False
        cause_start_position = 0
        cause_end_position = 0
        effect_start_position = 0
        effect_end_position = 0
        doc_start = span.data["start"]
        doc_end = span.data["start"] + span.data["length"] - 1
        out_of_span = False
        if tokenizer.padding_side == "left":
            doc_offset = 0
        else:
            doc_offset = sequence_added_tokens
        if tok_sentence_2_offset is not None:
            sentence_2_offset = tok_sentence_2_offset - doc_start + doc_offset
        else:
            sentence_2_offset = None
        if tok_sentence_3_offset is not None:
            sentence_3_offset = tok_sentence_3_offset - doc_start + doc_offset
        else:
            sentence_3_offset = None
        if is_training:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
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
                sentence_2_offset=sentence_2_offset,
                sentence_3_offset=sentence_3_offset,
                is_impossible=span_is_impossible,
            )
        )
    return features


def fincausal_convert_examples_to_features(examples: List[FinCausalExample],
                                           tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                                           max_seq_length: int,
                                           doc_stride: int,
                                           is_training: bool, return_dataset: Union[bool, str] = False,
                                           threads: int = 1) -> Union[List[FinCausalFeatures],
                                                                      Tuple[List[FinCausalFeatures], TensorDataset]]:
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
                position=0,
                leave=True
            )
        )
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(features, total=len(features), desc="add example index and unique id",
                                 position=0, leave=True):
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


def _run_split_on_punc(text: str) -> str:
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
