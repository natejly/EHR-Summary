# Usage
First, create an object from the LLM_as_a_judge_dpo class. The following arguments are required:
* input_data -> this can either be the path to a CSV file **OR** a pandas dataframe in memory, see Important note below
* output_dir -> the output directory where the DPO finetuned model will be stored
* model_id -> the path to the downloaded LLM model
* peft_model -> the path to the PEFT finetuned model

Important note for input_data argument:
* the dataframe or CSV file must be formatted with 3 columns: "prompt", "chosen", "rejected" all of which are strings 
* if you are reading in a CSV file, then set `input_data` to the file path of the CSV file, and set `read_csv_bool=True`
* if you are passing a pandas dataframe and not reading in a CSV file, then set `input_data` to the pandas dataframe, and set `read_csv_bool = False` or leave it as the default

Example usage for passing a pandas dataframe:

    llm_as_a_judge = LLM_as_a_judge_dpo(input_data=df, output_dir="dpo", model_id="llama3.1-8B", peft_model="peft_ft_model")

Example usage for reading a CSV file:

    llm_as_a_judge = LLM_as_a_judge_dpo(input_data="dpo_data.csv", output_dir="dpo", model_id="llama3.1-8B", peft_model="peft_ft_model", read_csv_bool=True)

Optional next step:
* If you wish to view the LoRA settings, run `print(llm_as_a_judge.qlora_config)`
  * If you wish to modify any of the LoRA settings, these are modifiable via the provided dataclass. For example, if you want to change `lora_alpha`, you'd run `llm_as_a_judge.qlora_config.lora_alpha = 12`
* If you wish to view the training arguments, run `print(llm_as_a_judge.training_args)`
  * If you wish to modify any of the training arguments, these are modifiable via the provided dataclass. For example, if you want to change the output directory name, you'd run `llm_as_a_judge.training_args.output_dir = output_dir_run2`

To start training, simply run `llm_as_a_judge.run_training()`

# Start to Finish example usage

    lm_as_a_judge = LLM_as_a_judge_dpo(input_data=df, output_dir="dpo", model_id="llama3.1-8B", peft_model="peft_ft_model")
    
    # change output directory name and number of training epochs
    llm_as_a_judge.training_args.num_train_epochs = 22
    llm_as_a_judge.training_args.output_dir = "output_dir_run1"
    
    # start training
    llm_as_a_judge.run_training()

# Additional
If you wish to add an argument for DPOConfig() that does has not been implemented, then add this to the Settings_Training_Args dataclass.
