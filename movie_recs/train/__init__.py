import numpy as np
import logging
import math
import os
import sys

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

import ray
from ray import tune
from ray.tune import CLIReporter

from ray.tune.examples.pbt_transformers.trainer import TuneTransformerTrainer

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def recover_checkpoint(tune_checkpoint_dir, model_name=None):
    if tune_checkpoint_dir is None or len(tune_checkpoint_dir) == 0:
        return model_name
    # Get subdirectory used for Huggingface.
    subdirs = [
        os.path.join(tune_checkpoint_dir, name)
        for name in os.listdir(tune_checkpoint_dir)
        if os.path.isdir(os.path.join(tune_checkpoint_dir, name))
    ]
    # There should only be 1 subdir.
    assert len(subdirs) == 1, subdirs
    return subdirs[0]


def define_data_classes():
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class ModelArguments:
        """
        Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
        """

        model_name_or_path: Optional[str] = field(
            default=None,
            metadata={
                "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
            },
        )
        model_type: Optional[str] = field(
            default=None,
            # metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
        )
        config_name: Optional[str] = field(
            default=None,
            metadata={
                "help": "Pretrained config name or path if not the same as model_name"
            })
        tokenizer_name: Optional[str] = field(
            default=None,
            metadata={
                "help": "Pretrained tokenizer name or path if not the same as model_name"
            })
        cache_dir: Optional[str] = field(
            default=None,
            metadata={
                "help": "Where do you want to store the pretrained models downloaded from s3"
            })

    @dataclass
    class DataTrainingArguments:
        """
        Arguments pertaining to what data we are going to input our model for training and eval.
        """

        train_data_file: Optional[str] = field(
            default=None,
            metadata={"help": "The input training data file (a text file)."})
        eval_data_file: Optional[str] = field(
            default=None,
            metadata={
                "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
            },
        )
        line_by_line: bool = field(
            default=False,
            metadata={
                "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
            },
        )

        mlm: bool = field(
            default=False,
            metadata={
                "help": "Train with masked-language modeling loss instead of language modeling."
            })
        mlm_probability: float = field(
            default=0.15,
            metadata={
                "help": "Ratio of tokens to mask for masked language modeling loss"
            })

        block_size: int = field(
            default=-1,
            metadata={
                "help": "Optional input sequence length after tokenization."
                "The training dataset will be truncated in block of this size for training."
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            },
        )
        overwrite_cache: bool = field(
            default=False,
            metadata={
                "help": "Overwrite the cached training and evaluation sets"
            })

    return ModelArguments, DataTrainingArguments


def get_dataset(args, tokenizer: PreTrainedTokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size)
    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size,
            overwrite_cache=args.overwrite_cache)


def load_and_validate_config(model_args, data_args):
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning(
            "You are instantiating a new config instance from scratch.")

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"
                             ] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling).")
    return config


def load_model(model_args, config):
    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)
    return model


def load_tokenizer(model_args):
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name")
    return tokenizer


def create_data_objects(data_args, training_args, tokenizer):
    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    train_dataset = get_dataset(
        data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(
        data_args, tokenizer=tokenizer,
        evaluate=True) if training_args.do_eval else None
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=data_args.mlm,
        mlm_probability=data_args.mlm_probability)
    return train_dataset, eval_dataset, data_collator


class CustomTrainer(TuneTransformerTrainer):
    def log(self, logs, **kwargs):
        super().log(logs, **kwargs)
        if np.isnan(logs["loss"]):
            logs["done"] = True
        tune.report(**logs)

    def _log(self, *args, **kwargs):
        pass


def train_language_model(config, checkpoint_dir=None):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    ModelArguments, DataTrainingArguments = define_data_classes()
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,
                               TrainingArguments))
    parsed = parser.parse_args_into_dataclasses(
        config["args"], return_remaining_strings=True)
    model_args, data_args, training_args = parsed[:-1]
    training_args.learning_rate = config["learning_rate"]
    training_args.weight_decay = config["weight_decay"]
    training_args.disable_tqdm = True
    logger.info("Training/evaluation parameters %s", training_args)

    tokenizer = load_tokenizer(model_args)

    train_dataset, eval_dataset, data_collator = create_data_objects(
        data_args, training_args, tokenizer)

    config = load_and_validate_config(model_args, data_args)
    model = load_model(model_args, config)

    model.resize_token_embeddings(len(tokenizer))

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )
    trainer.args.disable_tqdm = True

    model_path = (model_args.model_name_or_path
                  if model_args.model_name_or_path is not None
                  and os.path.isdir(model_args.model_name_or_path) else None)
    trainer.train(model_path=model_path)
    # model_path=recover_checkpoint(checkpoint_dir, config["model_name"]))


HUGGINNG_FACE_ARGS = [
    '--output_dir=bert-feat-out', '--model_type=distilbert',
    '--model_name_or_path=distilbert-base-uncased', '--mlm', '--do_train',
    '--train_data_file=/tmp/training_data.txt', '--block_size', '64',
    '--overwrite_output_dir', '--num_train_epochs', '5', '--save_total_limit',
    '1', '--save_steps=500', '--logging_steps=20', '--ray-address=auto',
    '--log-to-driver=False'
]


def huggingface_bert_trainer():
    # Download and cache tokenizer, model, and features
    ModelArguments, DataTrainingArguments = define_data_classes()
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,
                               TrainingArguments))
    parsed = parser.parse_args_into_dataclasses(
        HUGGINNG_FACE_ARGS, return_remaining_strings=True)
    model_args, data_args, training_args = parsed[:-1]

    tokenizer = load_tokenizer(model_args)
    create_data_objects(data_args, training_args, tokenizer)
    config = load_and_validate_config(model_args, data_args)
    load_model(model_args, config)

    ray.init(address="auto", log_to_driver=False, ignore_reinit_error=True)
    return train_language_model, HUGGINNG_FACE_ARGS
