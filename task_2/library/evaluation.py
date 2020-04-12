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
import os
import timeit
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm

from task_2.library.data import FinCausalResult
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

    # predictions = compute_predictions_logits(
    #     examples,
    #     features,
    #     all_results,
    #     n_best_size,
    #     max_answer_length,
    #     do_lower_case,
    #     output_prediction_file,
    #     output_nbest_file,
    #     output_null_log_odds_file,
    #     verbose_logging,
    #     False,
    #     null_score_diff_threshold,
    #     tokenizer,
    # )
    #
    # # Compute the F1 and exact scores.
    # results = squad_evaluate(examples, predictions)
    # return results


# ToDo: define strategy to create final results with 2 spans extracted
def compute_predictions_logits(
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        verbose_logging,
        version_2_with_negative,
        null_score_diff_threshold,
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

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index",
                             "start_cause_index", "end_cause_index", "start_cause_logit", "end_cause_logit",
                             "start_effect_index", "end_effect_index", "start_effect_logit", "end_effect_logit",
                             ]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            assert isinstance(result, FinCausalResult)
            start_cause_indexes = _get_best_indexes(result.start_cause_logits, n_best_size)
            end_cause_indexes = _get_best_indexes(result.end_cause_logits, n_best_size)
            start_effect_indexes = _get_best_indexes(result.start_effect_logits, n_best_size)
            end_effect_indexes = _get_best_indexes(result.end_effect_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            for start_index in start_cause_indexes:
                for end_index in end_cause_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
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
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                # tok_text = tok_text.replace(" ##", "")
                # tok_text = tok_text.replace("##", "")

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

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

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
