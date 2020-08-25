from typing import Dict

from library.config import ModelConfigurations, RunConfig


def initialize_log_dict(model_config: ModelConfigurations,
                        run_config: RunConfig,
                        model_tokenizer_mapping: Dict) -> Dict:
    model_type = model_config.value[0]
    model_class, tokenizer_class = model_tokenizer_mapping[model_type]

    return {'MODEL_TYPE': model_config.value[0],
            'MODEL_CLASS': model_class.__name__,
            'TOKENIZER_CLASS': tokenizer_class.__name__,
            'MODEL_NAME_OR_PATH': model_config.value[1],
            'DO_TRAIN': run_config.do_train,
            'DO_EVAL': run_config.do_eval,
            'DO_LOWER_CASE': model_config.value[2],
            'MAX_SEQ_LENGTH': run_config.max_seq_length,
            'DOC_STRIDE': run_config.doc_stride,
            'PER_GPU_BATCH_SIZE': run_config.train_batch_size,
            'GRADIENT_ACCUMULATION_STEPS': run_config.gradient_accumulation_steps,
            'WARMUP_STEPS': run_config.warmup_steps,
            'LEARNING_RATE': run_config.learning_rate,
            'NUM_TRAIN_EPOCHS': run_config.num_train_epochs,
            'PER_GPU_EVAL_BATCH_SIZE': run_config.eval_batch_size,
            'N_BEST_SIZE': run_config.n_best_size,
            'MAX_ANSWER_LENGTH': run_config.max_answer_length,
            'SENTENCE_BOUNDARY_HEURISTIC': run_config.sentence_boundary_heuristic,
            'FULL_SENTENCE_HEURISTIC': run_config.full_sentence_heuristic,
            'SHARED_SENTENCE_HEURISTIC': run_config.shared_sentence_heuristic,
            'OPTIMIZER': str(run_config.optimizer_class),
            'WEIGHT_DECAY': run_config.weight_decay,
            'SCHEDULER_FUNCTION': str(run_config.scheduler_function)
            }
