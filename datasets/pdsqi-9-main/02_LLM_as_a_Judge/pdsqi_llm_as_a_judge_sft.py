import os
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback
)
from datasets import load_dataset, ClassLabel, Value, Dataset, DatasetDict
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from trl import SFTTrainer
import gc
import numpy as np
from dataclasses import dataclass, field
from typing import List

class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.learning_rate = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
        if "learning_rate" in logs:
            self.learning_rate.append(logs["learning_rate"])

@dataclass
class Settings_QLORA_Config:
    r: int = 8
    lora_alpha: int = 8
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules : List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

@dataclass
class Settings_Training_Args:
    # training arguments
    learning_rate: float = 1e-07 #
    num_train_epochs: int = 100 #
    warmup_ratio : float = 0.1 #
    optim : str = "adamw_torch" # 
    # Batch & Gradient
    per_device_train_batch_size : int = 16 #Change as GPU resources allow #
    per_device_eval_batch_size : int = 16 #Change as GPU resources allow #
    gradient_accumulation_steps : int = 1 #Change as GPU resources allow #
    gradient_checkpointing : bool = True # 
    # Saving
    save_strategy : str = "epoch"
    output_dir : str = "output_dir", # CHANGEME
    # Logging
    logging_strategy : str = "epoch"
    logging_dir : str = "output_dir/logs" # CHANGEME
    do_eval : bool =  True,
    eval_strategy : str =  "epoch"
    load_best_model_at_end : bool = True #

class LLM_as_a_judge_sft:
    def __init__(self, input_data, valid_data, output_dir, model_id, peft_model):
        self.input_data = input_data # must be pandas dataframe or CSV file
        self.valid_data = valid_data # must be pandas dataframe or CSV file
        self.qlora_config = Settings_QLORA_Config()
        self.training_args = Settings_Training_Args()
        self.output_dir = output_dir
        self.model_id = model_id
        self.peft_model = peft_model
        self.read_csv_train_bool = False
        self.read_csv_valid_bool = False

    def run_training(self):
        read_csv_train_bool = self.read_csv_train_bool
        read_csv_valid_bool = self.read_csv_valid_bool
        qlora_config = self.qlora_config
        training_args = self.training_args

        # data setup
        # Data must be pandas df or CSV file formatted with a "prompt" and "completion" column, both of which are str
        training_dataset = None
        if read_csv_train_bool:
            training_dataset = pd.read_csv(self.input_data)
        else:
            training_dataset = self.input_data

        validation_dataset = None
        if read_csv_valid_bool:
            validation_dataset = pd.read_csv(self.valid_data)
        else:
            validation_dataset = self.valid_data
        
        dataset = DatasetDict({"train":training_dataset, "test": validation_dataset})

        # model and tokenizer setup
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant = True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                quantization_config = bnb_config,
                device_map = 'auto'
            )
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

        peft_config = LoraConfig(
                lora_alpha = qlora_config.lora_alpha,
                r = qlora_config.r,
                bias = qlora_config.bias,
                task_type = qlora_config.task_type,
                target_modules =  qlora_config.target_modules
            )
        model = get_peft_model(model, peft_config)

        # training setup
        torch.cuda.empty_cache()
        gc.collect()
        
        loss_logger = LossLoggerCallback()
        
        training_args = TrainingArguments(
                num_train_epochs = training_args.num_train_epochs,
                learning_rate = training_args.learning_rate,
                per_device_train_batch_size = training_args.batch_size,
                per_device_eval_batch_size = training_args.batch_size,
                gradient_accumulation_steps = training_args.gradient_accumulation_steps,
                logging_strategy = training_args.logging_strategy,
                save_strategy = training_args.save_strategy,
                output_dir = training_args.output_dir,
                logging_dir = training_args.logging_dir,
                optim = training_args.optim,
                warmup_ratio = training_args.warmup_ratio,
                do_eval = training_args.do_eval,
                eval_strategy = training_args.eval_strategy,
                gradient_checkpointing = training_args.gradient_checkpointing,
                load_best_model_at_end = training_args.load_best_model_at_end        
            )
        
        sft_trainer = SFTTrainer(
                model,
                args = training_args,
                train_dataset = dataset['train'],
                eval_dataset = dataset['test'],
                processing_class = tokenizer,
                callbacks = [loss_logger, EarlyStoppingCallback(early_stopping_patience = 3, early_stopping_threshold = 0.1)]
            )

        # run training
        print("Training Model...")
        torch.cuda.empty_cache()
        sft_trainer.train()
        
        # Save Model
        output_dir = os.path.join(self.output_dir, "final_checkpoint")
        print(f"Saving Pretrained Model at location {output_dir}")
        sft_trainer.model.save_pretrained(output_dir)
        print(f"Saving Pretrained Tokenizer at location {output_dir}")
        tokenizer.save_pretrained(output_dir)