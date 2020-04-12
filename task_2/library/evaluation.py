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

from task_2.library.data import FinCausalResult, FinCausalFeatures, FinCausalExample
from task_2.library.preprocessing import load_and_cache_examples

logger = logging.getLogger(__name__)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def evaluate(model, tokenizer, device: torch.device, file_path: Path, model_type: str, model_name_or_path: str,
             max_seq_length: int, doc_stride: int, eval_batch_size: int, output_dir: str,
             n_best_size: int, max_answer_length: int, do_lower_case: bool,
             null_score_diff_threshold: float = 0.0, verbose_logging: bool = False, prefix=""):
    dataset, examples, features = load_and_cache_examples(file_path, model_name_or_path, tokenizer,
                                                          max_seq_length, doc_stride,
                                                          output_examples=True, evaluate=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
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

            example_indices = batch[3]
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            start_cause_logits, end_cause_logits, start_effect_logits, end_effect_logits = output
            result = FinCausalResult(unique_id,
                                     start_cause_logits, end_cause_logits,
                                     start_effect_logits, end_effect_logits)

            all_results.append(result)

    eval_time = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", eval_time, eval_time / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(output_dir, "nbest_predictions_{}.json".format(prefix))

    output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        verbose_logging,
        tokenizer,
    )

    # # Compute the F1 and exact scores.
    # results = squad_evaluate(examples, predictions)
    # return results


class SpanType(Enum):
    cause = 0,
    effect = 1


_PrelimPrediction = collections.namedtuple(
    "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
)

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_logit", "end_logit"]
)


def filter_impossible_spans(features, span_type: SpanType, unique_id_to_result: Dict,
                            n_best_size: int, max_answer_length: int) -> List[_PrelimPrediction]:
    prelim_predictions = []

    for (feature_index, feature) in enumerate(features):
        result = unique_id_to_result[feature.unique_id]
        assert isinstance(result, FinCausalResult)
        assert isinstance(span_type, SpanType)
        if span_type == SpanType.cause:
            start_indexes = _get_best_indexes(result.start_cause_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_cause_logits, n_best_size)
            start_logits = result.start_cause_logits
            end_logits = result.end_cause_logits
        else:
            start_indexes = _get_best_indexes(result.start_effect_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_effect_logits, n_best_size)
            start_logits = result.start_effect_logits
            end_logits = result.end_effect_logits
        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > max_answer_length:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=start_logits[start_index],
                        end_logit=end_logits[end_index],
                    )
                )
    return prelim_predictions


def get_predictions(preliminary_predictions: List[_PrelimPrediction], n_best_size: int,
                    features: List[FinCausalFeatures], example: FinCausalExample,
                    tokenizer, do_lower_case: bool, verbose_logging: bool) -> List[_NbestPrediction]:
    seen_predictions = {}
    nbest = []
    for prediction in preliminary_predictions:
        if len(nbest) >= n_best_size:
            break
        feature = features[prediction.feature_index]
        if prediction.start_index > 0:  # this is a non-null prediction
            tok_tokens = feature.tokens[prediction.start_index: (prediction.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[prediction.start_index]
            orig_doc_end = feature.token_to_orig_map[prediction.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

            tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            final_text = ""
            seen_predictions[final_text] = True

        nbest.append(
            _NbestPrediction(text=final_text, start_logit=prediction.start_logit, end_logit=prediction.end_logit))
    return nbest


# ToDo: define strategy to create final results with 2 spans extracted simultaneously
# ToDo: check if the cause and effect can overlap. If not need to decode them at the same time and mark the overlaps
#  as impossible
def compute_predictions_logits(
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        verbose_logging,
        tokenizer,
):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % output_prediction_file)
    logger.info("Writing nbest to: %s" % output_nbest_file)

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        prelim_cause_predictions = filter_impossible_spans(features,
                                                           SpanType.cause,
                                                           unique_id_to_result,
                                                           n_best_size,
                                                           max_answer_length)
        prelim_effect_predictions = filter_impossible_spans(features,
                                                            SpanType.effect,
                                                            unique_id_to_result,
                                                            n_best_size,
                                                            max_answer_length)
        prelim_cause_predictions = sorted(prelim_cause_predictions, key=lambda x: (x.start_logit + x.end_logit),
                                          reverse=True)
        prelim_effect_predictions = sorted(prelim_effect_predictions, key=lambda x: (x.start_logit + x.end_logit),
                                           reverse=True)

        nbest_cause = get_predictions(prelim_cause_predictions, n_best_size, features, example, tokenizer,
                                      do_lower_case, verbose_logging)
        nbest_effect = get_predictions(prelim_effect_predictions, n_best_size, features, example, tokenizer,
                                       do_lower_case, verbose_logging)

        # In very rare edge cases we could have no valid predictions. So we
        # just create a none prediction in this case to avoid failure.
        if not nbest_cause:
            nbest_cause.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
        if not nbest_effect:
            nbest_effect.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest_cause) >= 1 and len(nbest_effect) >= 1

        total_cause_scores = []
        best_non_null_cause_entry = None
        for entry in nbest_cause:
            total_cause_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_cause_entry:
                if entry.text:
                    best_non_null_cause_entry = entry

        cause_probs = _compute_softmax(total_cause_scores)

        total_effect_scores = []
        best_non_null_effect_entry = None
        for entry in nbest_effect:
            total_effect_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_effect_entry:
                if entry.text:
                    best_non_null_effect_entry = entry

        effect_probs = _compute_softmax(total_effect_scores)

        nbest_json = []
        for (i, (cause_entry, effect_entry)) in enumerate(zip(nbest_cause, nbest_effect)):
            output = collections.OrderedDict()
            output["cause_text"] = cause_entry.text
            output["cause_probability"] = cause_probs[i]
            output["cause_start_logit"] = cause_entry.start_logit
            output["cause_end_logit"] = cause_entry.end_logit
            output["effect_text"] = effect_entry.text
            output["effect_probability"] = effect_probs[i]
            output["effect_start_logit"] = effect_entry.start_logit
            output["effect_end_logit"] = effect_entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.example_id] = {"cause_text": nbest_json[0]["cause_text"],
                                               "effect_text": nbest_json[0]["effect_text"]}
        all_nbest_json[example.example_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    return all_predictions


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return ns_text, ns_to_s_map

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position: (orig_end_position + 1)]
    return output_text


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
