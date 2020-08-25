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

import logging
import os
from pathlib import Path
from typing import Dict

import torch
from torch.nn import Module
from torch.utils.data import RandomSampler, DataLoader, TensorDataset
from tqdm import trange, tqdm
from transformers import PreTrainedTokenizerBase

from .config import RunConfig
from .evaluation import evaluate

logger = logging.getLogger(__name__)


def train(train_dataset: TensorDataset,
          model: Module,
          tokenizer: PreTrainedTokenizerBase,
          model_type: str,
          model_name_or_path: str,
          output_dir: str,
          predict_file: Path,
          log_file: Dict,
          device: torch.device,
          evaluate_during_training: bool,
          run_config: RunConfig):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=run_config.train_batch_size)

    t_total = len(train_dataloader) // run_config.gradient_accumulation_steps * run_config.num_train_epochs

    # Define Optimizer and learning rates / decay
    no_decay = ["bias", "LayerNorm.weight"]
    no_scaled_lr = ["cause_outputs", "effect_outputs"]
    if run_config.differential_lr_ratio == 0:
        differential_lr_ratio = 1.0
    else:
        differential_lr_ratio = run_config.differential_lr_ratio
    assert differential_lr_ratio <= 1, "ratio for language model layers should be <= 1"
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
                                                                  and not any(nlr in n for nlr in no_scaled_lr))],
            'lr': run_config.learning_rate * differential_lr_ratio,
            "weight_decay": run_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)
                                                                  and any(nlr in n for nlr in no_scaled_lr))],
            'lr': run_config.learning_rate,
            "weight_decay": run_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
                                                                  and not any(nlr in n for nlr in no_scaled_lr))],
            'lr': run_config.learning_rate * differential_lr_ratio,
            "weight_decay": 0.0
        },
        {
            "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)
                                                                  and any(nlr in n for nlr in no_scaled_lr))],
            'lr': run_config.learning_rate,
            "weight_decay": 0.0
        },
    ]
    optimizer = run_config.optimizer_class(optimizer_grouped_parameters,
                                           lr=run_config.learning_rate,
                                           eps=run_config.adam_epsilon)

    # Define Scheduler
    try:
        scheduler = run_config.scheduler_function(optimizer,
                                                  num_warmup_steps=run_config.warmup_steps,
                                                  num_training_steps=t_total)
    except ValueError:
        scheduler = run_config.scheduler_function(optimizer,
                                                  num_warmup_steps=run_config.warmup_steps)

    # Start training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", run_config.num_train_epochs)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        run_config.train_batch_size * run_config.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", run_config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(run_config.num_train_epochs), desc="Epoch")

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc=f"Iteration Loss: {tr_loss / global_step}", position=0, leave=True)
        for step, batch in enumerate(epoch_iterator):
            epoch_iterator.set_description(f"Iteration Loss: {tr_loss / global_step}")

            model.train()
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_cause_positions": batch[3],
                "end_cause_positions": batch[4],
                "start_effect_positions": batch[5],
                "end_effect_positions": batch[6],
            }

            if model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            outputs = model(**inputs)
            loss = outputs[0]

            if run_config.gradient_accumulation_steps > 1:
                loss = loss / run_config.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % run_config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), run_config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

        # Log metrics
        if evaluate_during_training:
            metrics = evaluate(model=model,
                               tokenizer=tokenizer,
                               device=device,
                               file_path=predict_file,
                               model_type=model_type,
                               model_name_or_path=model_name_or_path,
                               max_seq_length=run_config.max_seq_length,
                               doc_stride=run_config.doc_stride,
                               eval_batch_size=run_config.eval_batch_size,
                               output_dir=output_dir,
                               n_best_size=run_config.n_best_size,
                               max_answer_length=run_config.max_answer_length,
                               sentence_boundary_heuristic=run_config.sentence_boundary_heuristic,
                               full_sentence_heuristic=run_config.full_sentence_heuristic,
                               shared_sentence_heuristic=run_config.shared_sentence_heuristic,
                               top_n_sentences=run_config.top_n_sentences)
            log_file[f'step_{global_step}'] = metrics

            _output_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
            if not os.path.exists(_output_dir):
                os.makedirs(_output_dir)

            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(_output_dir)
            tokenizer.save_pretrained(_output_dir)
            logger.info("Best F1 score: saving model checkpoint to %s", _output_dir)

    return global_step, tr_loss / global_step
