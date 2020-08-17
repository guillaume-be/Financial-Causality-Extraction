# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2020 Guillaume Becquin.
# MODIFIED FOR CAUSE EFFECT EXTRACTION
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

import unicodedata
from typing import Optional, List

import pandas as pd


class FinCausalExample:

    def __init__(
            self,
            example_id: str,
            context_text: str,
            cause_text: str,
            effect_text: str,
            offset_sentence_2: int,
            offset_sentence_3: int,
            cause_start_position_character: Optional[int],
            cause_end_position_character: Optional[int],
            effect_start_position_character: Optional[int],
            effect_end_position_character: Optional[int]
    ):

        self.example_id = example_id
        self.context_text = context_text
        self.cause_text = cause_text
        self.effect_text = effect_text

        self.start_cause_position, self.end_cause_position = 0, 0
        self.start_effect_position, self.end_effect_position = 0, 0

        doc_tokens: List[str] = []
        char_to_word_offset: List[int] = []
        word_to_char_mapping: List[int] = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for char_index, c in enumerate(self.context_text):
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace or _is_punctuation(c):
                    doc_tokens.append(c)
                    word_to_char_mapping.append(char_index)
                else:
                    doc_tokens[-1] += c
                if _is_punctuation(c):
                    prev_is_whitespace = True
                else:
                    prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if cause_start_position_character is not None:
            assert (cause_start_position_character + len(cause_text) == cause_end_position_character)
            self.start_cause_position = char_to_word_offset[cause_start_position_character]
            self.end_cause_position = char_to_word_offset[
                min(cause_start_position_character + len(cause_text) - 1, len(char_to_word_offset) - 1)
            ]
        if effect_start_position_character is not None:
            self.start_effect_position = char_to_word_offset[effect_start_position_character]
            assert (effect_start_position_character + len(effect_text) == effect_end_position_character)
            self.end_effect_position = char_to_word_offset[
                min(effect_start_position_character + len(effect_text) - 1, len(char_to_word_offset) - 1)
            ]

        if pd.notna(offset_sentence_2):
            self.offset_sentence_2 = char_to_word_offset[int(offset_sentence_2)]
        else:
            self.offset_sentence_2 = 0
        if pd.notna(offset_sentence_3):
            self.offset_sentence_3 = char_to_word_offset[int(offset_sentence_3)]
        else:
            self.offset_sentence_3 = 0

        self.word_to_char_mapping = word_to_char_mapping


class FinCausalFeatures:

    def __init__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            cls_index,
            p_mask,
            example_orig_index,
            example_index,
            unique_id,
            paragraph_len,
            token_is_max_context,
            tokens,
            token_to_orig_map,
            cause_start_position,
            cause_end_position,
            effect_start_position,
            effect_end_position,
            sentence_2_offset,
            sentence_3_offset,
            is_impossible,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_orig_index = example_orig_index
        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.cause_start_position = cause_start_position
        self.cause_end_position = cause_end_position
        self.effect_start_position = effect_start_position
        self.effect_end_position = effect_end_position

        self.sentence_2_offset = sentence_2_offset
        self.sentence_3_offset = sentence_3_offset

        self.is_impossible = is_impossible


class FinCausalResult:
    def __init__(self, unique_id, start_cause_logits, end_cause_logits, start_effect_logits, end_effect_logits,
                 start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_cause_logits = start_cause_logits
        self.end_cause_logits = end_cause_logits
        self.start_effect_logits = start_effect_logits
        self.end_effect_logits = end_effect_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or c == '\xa0' or ord(c) == 0x202F:
        return True
    return False


def _is_punctuation(char):
    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
