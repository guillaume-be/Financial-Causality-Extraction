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


def evaluate_one_step(model, tokenizer, device: torch.device, file_path: Path, model_type: str, model_name_or_path: str,
                      max_seq_length: int, doc_stride: int, eval_batch_size: int, output_dir: str,
                      n_best_size: int, max_answer_length: int, do_lower_case: bool,
                      verbose_logging: bool = False, prefix=""):
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
    "PrelimPrediction", ["feature_index",
                         "start_index_cause", "end_index_cause", "start_logit_cause", "end_logit_cause",
                         "start_index_effect", "end_index_effect", "start_logit_effect", "end_logit_effect"]
)

_NbestPrediction = collections.namedtuple(
    "NbestPrediction", ["text_cause", "start_logit_cause", "end_logit_cause",
                        "text_effect", "start_logit_effect", "end_logit_effect"]
)


def filter_impossible_spans(features, unique_id_to_result: Dict,
                            n_best_size: int, max_answer_length: int) -> List[_PrelimPrediction]:
    prelim_predictions = []

    for (feature_index, feature) in enumerate(features):
        result = unique_id_to_result[feature.unique_id]
        assert isinstance(result, FinCausalResult)
        start_indexes_cause = _get_best_indexes(result.start_cause_logits, n_best_size)
        end_indexes_cause = _get_best_indexes(result.end_cause_logits, n_best_size)
        start_logits_cause = result.start_cause_logits
        end_logits_cause = result.end_cause_logits
        start_indexes_effect = _get_best_indexes(result.start_effect_logits, n_best_size)
        end_indexes_effect = _get_best_indexes(result.end_effect_logits, n_best_size)
        start_logits_effect = result.start_effect_logits
        end_logits_effect = result.end_effect_logits

        for start_index_cause in start_indexes_cause:
            for start_index_effect in start_indexes_effect:
                for end_index_cause in end_indexes_cause:
                    for end_index_effect in end_indexes_effect:
                        if (start_index_cause <= start_index_effect) and (end_index_cause >= start_index_effect):
                            continue
                        if (start_index_effect <= start_index_cause) and (end_index_effect >= start_index_cause):
                            continue
                        if start_index_effect >= len(feature.tokens) or start_index_cause >= len(feature.tokens):
                            continue
                        if end_index_effect >= len(feature.tokens) or end_index_cause >= len(feature.tokens):
                            continue
                        if start_index_effect not in feature.token_to_orig_map or start_index_cause not in feature.token_to_orig_map:
                            continue
                        if end_index_effect not in feature.token_to_orig_map or end_index_cause not in feature.token_to_orig_map:
                            continue
                        if (not feature.token_is_max_context.get(start_index_effect, False)) or \
                                (not feature.token_is_max_context.get(start_index_cause, False)):
                            continue
                        if end_index_cause < start_index_cause:
                            continue
                        if end_index_effect < start_index_effect:
                            continue
                        length_cause = end_index_cause - start_index_cause + 1
                        length_effect = end_index_effect - start_index_effect + 1
                        if length_cause > max_answer_length:
                            continue
                        if length_effect > max_answer_length:
                            continue

                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index_cause=start_index_cause,
                                end_index_cause=end_index_cause,
                                start_logit_cause=start_logits_cause[start_index_cause],
                                end_logit_cause=end_logits_cause[end_index_cause],
                                start_index_effect=start_index_effect,
                                end_index_effect=end_index_effect,
                                start_logit_effect=start_logits_effect[start_index_effect],
                                end_logit_effect=end_logits_effect[end_index_effect]
                            )
                        )
    return prelim_predictions


def get_predictions(preliminary_predictions: List[_PrelimPrediction], n_best_size: int,
                    features: List[FinCausalFeatures], example: FinCausalExample,
                    tokenizer, do_lower_case: bool, verbose_logging: bool) -> List[_NbestPrediction]:
    seen_predictions_cause = {}
    seen_predictions_effect = {}
    nbest = []
    for prediction in preliminary_predictions:
        if len(nbest) >= n_best_size:
            break
        feature = features[prediction.feature_index]
        if prediction.start_index_cause > 0:  # this is a non-null prediction
            tok_tokens_cause = feature.tokens[prediction.start_index_cause: (prediction.end_index_cause + 1)]
            orig_doc_start_cause = feature.token_to_orig_map[prediction.start_index_cause]
            orig_doc_end_cause = feature.token_to_orig_map[prediction.end_index_cause]
            orig_tokens_cause = example.doc_tokens[orig_doc_start_cause: (orig_doc_end_cause + 1)]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens_cause)
            # Clean whitespace
            tok_text_cause = tok_text.strip()
            tok_text_cause = " ".join(tok_text_cause.split())
            orig_text_cause = " ".join(orig_tokens_cause)
            final_text_cause = get_final_text(tok_text_cause, orig_text_cause, do_lower_case, verbose_logging)

            tok_tokens_effect = feature.tokens[prediction.start_index_effect: (prediction.end_index_effect + 1)]
            orig_doc_start_effect = feature.token_to_orig_map[prediction.start_index_effect]
            orig_doc_end_effect = feature.token_to_orig_map[prediction.end_index_effect]
            orig_tokens_effect = example.doc_tokens[orig_doc_start_effect: (orig_doc_end_effect + 1)]
            tok_text = tokenizer.convert_tokens_to_string(tok_tokens_effect)
            # Clean whitespace
            tok_text_effect = tok_text.strip()
            tok_text_effect = " ".join(tok_text_effect.split())
            orig_text_effect = " ".join(orig_tokens_effect)
            final_text_effect = get_final_text(tok_text_effect, orig_text_effect, do_lower_case, verbose_logging)

            if final_text_cause in seen_predictions_cause and final_text_effect in seen_predictions_effect:
                continue

            seen_predictions_cause[final_text_cause] = True
            seen_predictions_cause[final_text_effect] = True
        else:
            final_text_cause = final_text_effect = ""
            seen_predictions_cause[final_text_cause] = True
            seen_predictions_cause[final_text_effect] = True

        nbest.append(
            _NbestPrediction(text_cause=final_text_cause, start_logit_cause=prediction.start_index_cause,
                             end_logit_cause=prediction.end_index_cause,
                             text_effect=final_text_effect, start_logit_effect=prediction.start_index_effect,
                             end_logit_effect=prediction.end_index_effect))
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

        prelim_predictions = filter_impossible_spans(features,
                                                     unique_id_to_result,
                                                     n_best_size,
                                                     max_answer_length)
        prelim_predictions = sorted(prelim_predictions,
                                    key=lambda x: (x.start_logit_cause + x.end_logit_cause +
                                                   x.start_logit_effect + x.end_logit_effect),
                                    reverse=True)

        nbest = get_predictions(prelim_predictions, n_best_size, features, example, tokenizer,
                                do_lower_case, verbose_logging)

        # In very rare edge cases we could have no valid predictions. So we
        # just create a none prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text_cause="empty", start_logit_cause=0.0, end_logit_cause=0.0,
                                          text_effect="empty", start_logit_effect=0.0, end_logit_effect=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit_cause + entry.end_logit_cause +
                                entry.start_logit_effect + entry.end_logit_effect)
            if not best_non_null_entry:
                if entry.text_cause and entry.text_effect:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["probability"] = probs[i]
            output["cause_text"] = entry.text_cause
            output["cause_start_logit"] = entry.start_logit_cause
            output["cause_end_logit"] = entry.end_logit_cause
            output["effect_text"] = entry.text_effect
            output["effect_start_logit"] = entry.start_logit_effect
            output["effect_end_logit"] = entry.end_logit_effect
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
