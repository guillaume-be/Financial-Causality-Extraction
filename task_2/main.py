# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2020 Guillaume Becquin. All rights reserved.
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


import json
import logging
import os
from pathlib import Path
import torch
from library.config import RunConfig, ModelConfigurations, model_tokenizer_mapping
from library.evaluation import evaluate, predict
from library.logging import initialize_log_dict
from library.preprocessing import load_and_cache_examples
from library.training import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = ModelConfigurations.RoBERTaSquadLarge
    run_config = RunConfig()

    RUN_NAME = 'model_run'

    (MODEL_TYPE, MODEL_NAME_OR_PATH, DO_LOWER_CASE) = model_config.value
    fincausal_data_path = Path(os.environ['FINCAUSAL_DATA_PATH'])
    fincausal_output_path = Path(os.environ['FINCAUSAL_OUTPUT_PATH'])
    TRAIN_FILE = fincausal_data_path / "fnp2020-train.csv"
    EVAL_FILE = fincausal_data_path / "fnp2020-eval.csv"
    TEST_FILE = fincausal_data_path / "task2.csv"

    if RUN_NAME:
        OUTPUT_DIR = str(fincausal_output_path / (MODEL_NAME_OR_PATH + '_' + RUN_NAME))
    else:
        OUTPUT_DIR = str(fincausal_output_path / MODEL_NAME_OR_PATH)

    model_class, tokenizer_class = model_tokenizer_mapping[MODEL_TYPE]
    log_file = initialize_log_dict(model_config=model_config,
                                   run_config=run_config,
                                   model_tokenizer_mapping=model_tokenizer_mapping)

    # Training
    if run_config.do_train:

        tokenizer = tokenizer_class.from_pretrained(MODEL_NAME_OR_PATH,
                                                    do_lower_case=DO_LOWER_CASE,
                                                    cache_dir=OUTPUT_DIR)
        model = model_class.from_pretrained(MODEL_NAME_OR_PATH).to(device)

        train_dataset = load_and_cache_examples(file_path=TRAIN_FILE,
                                                tokenizer=tokenizer,
                                                output_examples=False,
                                                run_config=run_config)

        global_step, tr_loss = train(train_dataset=train_dataset,
                                     model=model,
                                     tokenizer=tokenizer,
                                     model_type=MODEL_TYPE,
                                     model_name_or_path=MODEL_NAME_OR_PATH,
                                     output_dir=OUTPUT_DIR,
                                     predict_file=EVAL_FILE,
                                     device=device,
                                     evaluate_during_training=True,
                                     log_file=log_file,
                                     )

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        if SAVE_MODEL:
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            logger.info("Saving final model to %s", OUTPUT_DIR)
        logger.info("Saving log file to %s", OUTPUT_DIR)
        with open(os.path.join(OUTPUT_DIR, "logs.json"), 'w') as f:
            json.dump(log_file, f, indent=4)

    if DO_EVAL:
        tokenizer = tokenizer_class.from_pretrained(OUTPUT_DIR, do_lower_case=DO_LOWER_CASE)
        model = model_class.from_pretrained(OUTPUT_DIR).to(device)

        result = evaluate(model=model,
                          tokenizer=tokenizer,
                          device=device,
                          file_path=EVAL_FILE,
                          model_type=MODEL_TYPE,
                          model_name_or_path=MODEL_NAME_OR_PATH,
                          max_seq_length=MAX_SEQ_LENGTH,
                          doc_stride=DOC_STRIDE,
                          eval_batch_size=PER_GPU_EVAL_BATCH_SIZE,
                          output_dir=OUTPUT_DIR,
                          n_best_size=N_BEST_SIZE,
                          max_answer_length=MAX_ANSWER_LENGTH,
                          sentence_boundary_heuristic=SENTENCE_BOUNDARY_HEURISTIC,
                          full_sentence_heuristic=FULL_SENTENCE_HEURISTIC,
                          shared_sentence_heuristic=SHARED_SENTENCE_HEURISTIC,
                          overwrite_cache=OVERWRITE_CACHE,
                          top_n_sentences=TOP_N_SENTENCES
                          )

        print("done")

    if DO_TEST:
        tokenizer = tokenizer_class.from_pretrained(OUTPUT_DIR, do_lower_case=DO_LOWER_CASE)
        model = model_class.from_pretrained(OUTPUT_DIR).to(device)

        result = predict(model=model,
                         tokenizer=tokenizer,
                         device=device,
                         file_path=TEST_FILE,
                         model_type=MODEL_TYPE,
                         model_name_or_path=MODEL_NAME_OR_PATH,
                         max_seq_length=MAX_SEQ_LENGTH,
                         doc_stride=DOC_STRIDE,
                         eval_batch_size=PER_GPU_EVAL_BATCH_SIZE,
                         output_dir=OUTPUT_DIR,
                         n_best_size=N_BEST_SIZE,
                         max_answer_length=MAX_ANSWER_LENGTH,
                         sentence_boundary_heuristic=SENTENCE_BOUNDARY_HEURISTIC,
                         full_sentence_heuristic=FULL_SENTENCE_HEURISTIC,
                         shared_sentence_heuristic=SHARED_SENTENCE_HEURISTIC,
                         overwrite_cache=OVERWRITE_CACHE,
                         top_n_sentences=TOP_N_SENTENCES
                         )

        print("done")
