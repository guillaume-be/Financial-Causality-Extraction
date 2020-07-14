from enum import Enum
from pathlib import Path
import json
from typing import Dict
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class ModelConfigurations(Enum):
    BertSquad = ('bert', 'deepset/bert-base-cased-squad2', False)
    BertSquad2 = ('bert', 'deepset/bert-large-uncased-whole-word-masking-squad2', True)
    DistilBertSquad = ('distilbert', 'distilbert-base-uncased-distilled-squad', True)
    RoBERTaSquad = ('roberta', 'deepset/roberta-base-squad2', False)
    RoBERTaSquadLarge = ('roberta', 'ahotrod/roberta_large_squad2', False)
    XLNetBase = ('xlnet', 'xlnet-base-cased', False)
    AlbertSquad = ('albert', 'twmkn9/albert-base-v2-squad2', True)


model_config = ModelConfigurations.RoBERTaSquadLarge
RUN_NAME = 'FULL_90pc_TRAIN_EVAL_2f5caf4c3c7866711f6a90b1bf69fe4744eb256c24c4ee4a0ea9fcaa8f2a4f25'

(_, MODEL_NAME_OR_PATH, _) = model_config.value
output_dir = Path('E:/Coding/finNLP/task_2/output') / (MODEL_NAME_OR_PATH + '_' + RUN_NAME)
nbest_file_valid = output_dir / 'nbest_predictions_valid.json'
nbest_file_test = output_dir / 'nbest_predictions_test.json'


def get_prediction_probability(prediction: Dict):
    prob_cause_start = _logit_to_proba(prediction['cause_start_score'])
    prob_cause_end = _logit_to_proba(prediction['cause_end_score'])
    prob_effect_start = _logit_to_proba(prediction['effect_start_score'])
    prob_effect_end = _logit_to_proba(prediction['effect_end_score'])
    return prob_cause_start * prob_cause_end * prob_effect_start * prob_effect_end


def _logit_to_proba(logit: float) -> float:
    odd = math.exp(logit)
    return odd / (odd + 1)


def get_threshold_f1(valid_scores: np.ndarray, invalid_scores: np.ndarray, threshold: float):
    tp = np.count_nonzero(valid_scores >= threshold)
    fn = np.count_nonzero(valid_scores < threshold)
    fp = np.count_nonzero(invalid_scores >= threshold)
    if tp > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = precision = recall = 0
    return f1, precision, recall


def get_max_f1(valid_scores: np.ndarray, invalid_scores: np.ndarray, step_size: float = 0.01):
    max_f1 = 0
    precision = 0
    recall = 0
    max_threshold = None
    for threshold in np.arange(0, 1 + step_size, step_size):
        new_scores = get_threshold_f1(valid_scores, invalid_scores, threshold)
        if new_scores[0] > max_f1:
            max_threshold = threshold
            precision = new_scores[1]
            recall = new_scores[2]
            max_f1 = new_scores[0]
    return max_f1, precision, recall, max_threshold


if __name__ == '__main__':
    with nbest_file_valid.open('r') as f:
        nbest_dict_valid = json.load(f)

    with nbest_file_test.open('r') as f:
        nbest_dict_test = json.load(f)

    example_id_num_predictions = dict()
    example_id_values = dict()
    for example_id, example_values in nbest_dict_valid.items():
        suffix_index = 1
        if example_id.count('.') == 2:
            suffix_index = int(example_id.split('.')[-1])
        example_id_num_predictions['.'.join(example_id.split('.')[:-1])] = suffix_index
        example_id_values['.'.join(example_id.split('.')[:-1])] = example_values

    for example_id, example_values in nbest_dict_test.items():
        suffix_index = 1
        if example_id.count('.') == 2:
            suffix_index = int(example_id.split('.')[-1])
        example_id_num_predictions['.'.join(example_id.split('.')[:-1])] = suffix_index
        example_id_values['.'.join(example_id.split('.')[:-1])] = example_values

    valid_multiple_prediction_scores = []
    invalid_multiple_prediction_scores = []

    for example_id in example_id_num_predictions.keys():
        num_predictions = example_id_num_predictions[example_id]
        predictions = example_id_values[example_id]

        for prediction_index, prediction in enumerate(predictions):
            if prediction['is_new'] and prediction_index > 0:
                if prediction_index < num_predictions:
                    valid_multiple_prediction_scores.append(get_prediction_probability(prediction))
                else:
                    invalid_multiple_prediction_scores.append(get_prediction_probability(prediction))

    valid_multiple_prediction_scores = np.array(valid_multiple_prediction_scores)
    invalid_multiple_prediction_scores = np.array(invalid_multiple_prediction_scores)

    print(get_max_f1(valid_multiple_prediction_scores, invalid_multiple_prediction_scores))

    sns.set_palette([[0.2, 0.2, 0.2, 1.], [0.884375, 0.5265625, 0, 1.]])
    sns.set_context("paper", rc={"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 50}, font_scale=1.3)
    plt.figure(figsize=(8, 5))
    bins = np.arange(0, 1.02, 0.02)
    sns.distplot(valid_multiple_prediction_scores, label="Valid predictions scores", kde=False, bins=bins)
    sns.distplot(invalid_multiple_prediction_scores, label="Invalid predictions scores", kde=False, bins=bins)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.savefig(output_dir / 'nbest_prediction', dpi=300)
