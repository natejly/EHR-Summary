import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
from datasets import load_dataset, ClassLabel, Value, Dataset
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)
from trl import DPOTrainer, DPOConfig
from dataclasses import dataclass, field
from typing import List

class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.reward_margin = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
        if "eval_rewards/margins" in logs:
            self.reward_margin.append(logs["eval_rewards/margins"])

@dataclass
class Settings_QLORA_Config:
    r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules : List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

@dataclass
class Settings_Training_Args:
    # training arguments
    learning_rate: float = 1e-07
    beta: float = 0.7
    num_train_epochs: int = 20
    warmup_ratio : float = 0.1
    optim : str = "adamw_torch"
    # Batch & Gradient
    per_device_train_batch_size : int = 1 #Change as GPU resources allow
    per_device_eval_batch_size : int = 1 #Change as GPU resources allow
    gradient_accumulation_steps : int = 1 #Change as GPU resources allow
    gradient_checkpointing : bool = True
    # Saving
    save_strategy : str = "steps"
    output_dir : str = "output_dir", # CHANGEME
    save_steps : int = 10
    # Logging
    logging_strategy : str = "steps"
    logging_dir : str = "output_dir/logs" # CHANGEME
    logging_steps : int = 10
    # Eval
    do_eval : bool =  True,
    eval_strategy : str =  "steps"
    eval_steps : int = 10
    # Callback
    load_best_model_at_end : bool = True
    metric_for_best_model : str = "eval_rewards/margins" # NB: must match to LossLoggerCallBack.on_log()
    greater_is_better : bool = False
    # DPO Specific
    max_length : int = 13768 #Change if needed
    max_prompt_length : int = 12768 #Change if needed   

class LLM_as_a_judge_dpo:
    def __init__(self, input_data, output_dir, model_id, peft_model):
        self.input_data = input_data # must be pandas dataframe
        self.qlora_config = Settings_QLORA_Config()
        self.training_args = Settings_Training_Args()
        self.output_dir = output_dir
        self.model_id = model_id
        self.peft_model = peft_model
        self.read_csv_bool = False

    def run_training(self):
        read_csv_bool = self.read_csv_bool   
        qlora_config = self.qlora_config
        training_args = self.training_args
        
        # data setup
        # Data should be pandas df or CSV file formatted with 3 columns: "prompt", "chosen", "rejected" all of which are strings
        training_data = None
        if read_csv_bool:
            training_data = pd.read_csv(self.input_data)
        else:
            training_data = self.input_data
        
        dataset = Dataset.from_pandas(training_data)
        dataset = dataset.train_test_split(test_size = 0.2)

        # setup model and tokenizer
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
                attn_implementation = "flash_attention_2",
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                quantization_config = bnb_config,
                device_map = 'auto'
            )
        
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

        model = PeftModel.from_pretrained(model, self.peft_model)
        model = model.merge_and_unload()

        peft_config = LoraConfig(
                lora_alpha = qlora_config.lora_alpha,
                lora_dropout = qlora_config.lora_dropout,
                r = qlora_config.r,
                bias = qlora_config.bias,
                task_type = qlora_config.task_type,
                target_modules =  qlora_config.target_modules
        )
        
        model = get_peft_model(model, peft_config)

        # training
        loss_logger = LossLoggerCallback()
        
        training_args = DPOConfig(
                #Base Training Stuff
                num_train_epochs = training_args.num_train_epochs, #Doesn't take this many, but trains so slow that it was usually manually stopped
                beta = training_args.beta,
                learning_rate = training_args.learning_rate,
                warmup_ratio = training_args.warmup_ratio,
                optim = training_args.optim,
                #Batch & Gradient
                per_device_train_batch_size = training_args.per_device_train_batch_size,
                per_device_eval_batch_size = training_args.per_device_eval_batch_size,
                gradient_accumulation_steps = training_args.gradient_accumulation_steps,
                gradient_checkpointing = training_args.gradient_checkpointing,
                #Saving
                save_strategy= training_args.save_strategy,
                output_dir = training_args.output_dir,
                save_steps = training_args.save_steps,
                #Logging
                logging_strategy = training_args.logging_strategy,
                logging_dir = training_args.logging_dir,
                logging_steps = training_args.logging_steps,
                #Eval
                do_eval = training_args.do_eval,
                eval_strategy= training_args.eval_strategy,
                eval_steps = training_args.eval_steps,
                #Callback
                load_best_model_at_end = training_args.load_best_model_at_end,
                metric_for_best_model = training_args.metric_for_best_model,
                greater_is_better = training_args.greater_is_better,
                #DPO Specific
                max_length = training_args.max_length, 
                max_prompt_length = training_args.max_prompt_length
        )

        dpo_trainer = DPOTrainer(
                model,
                ref_model = None,
                args = training_args,
                train_dataset = dataset['train'],
                eval_dataset = dataset['test'],
                tokenizer = tokenizer,
                callbacks = [loss_logger]
        
            )
        
        print("Training Model...")
        torch.cuda.empty_cache()
        dpo_trainer.train()
        
        #Save Stuff
        output_dir = os.path.join(self.output_dir, "final_checkpoint")
        print(f"Saving Pretrained Model at directory {output_dir}")
        dpo_trainer.model.save_pretrained(output_dir)
        print(f"Saving Pretrained Tokenizer at directory {output_dir}")
        tokenizer.save_pretrained(output_dir)