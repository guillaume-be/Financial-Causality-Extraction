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

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from task_2.library.evaluation import evaluate

logger = logging.getLogger(__name__)


def train(train_dataset, model, tokenizer, train_batch_size: int,
          model_type: str, model_name_or_path: str, output_dir: str, predict_file: Path, device: torch.device,
          max_steps: Optional[int], gradient_accumulation_steps: int, num_train_epochs: int, warmup_steps: int,
          logging_steps: int, save_steps: int, evaluate_during_training: bool,
          max_seq_length: int, doc_stride: int, eval_batch_size: int,
          n_best_size: int, max_answer_length: int, do_lower_case: bool,
          learning_rate: float, weight_decay: float = 0.0, adam_epsilon: float = 1e-8, max_grad_norm: float = 1.0):
    """ Train the model """
    tb_writer = SummaryWriter()

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    if max_steps is not None:
        t_total = max_steps
        num_train_epochs = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        train_batch_size * gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch")

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

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
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if (logging_steps > 0 and global_step % logging_steps == 0) or \
                        global_step % (len(train_dataloader) // gradient_accumulation_steps) == 0:
                    if evaluate_during_training:
                        results = evaluate(model, tokenizer, device, predict_file, model_type, model_name_or_path,
                                           max_seq_length, doc_stride, eval_batch_size, output_dir,
                                           n_best_size, max_answer_length, do_lower_case)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if (save_steps > 0 and global_step % save_steps == 0) or \
                        global_step % (len(train_dataloader) // gradient_accumulation_steps) == 0:
                    _output_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(_output_dir):
                        os.makedirs(_output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(_output_dir)
                    tokenizer.save_pretrained(_output_dir)

                    logger.info("Saving model checkpoint to %s", _output_dir)

                    # torch.save(optimizer.state_dict(), os.path.join(_output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(_output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", _output_dir)

            if max_steps is not None and global_step > max_steps:
                epoch_iterator.close()
                break
        if max_steps is not None and global_step > max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step
