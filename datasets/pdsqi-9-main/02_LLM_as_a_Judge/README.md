# Automated Evaluation using the PDSQI-9

We recommend using **GPT-03-mini** in a **Single LLM-as-a-Judge** workflow for best results.

## ⚖️ Single LLM-as-a-Judge Workflow

### 🔧 Option 1: Using a Local Model

1. Ensure your model is compatible with Hugging Face Transformers.
2. Download the notebook: [Single LLM-as-a-Judge (Local Model)](02_LLM_as_a_Judge/single_llm_as_a_judge_local.ipynb).
3. Install the required packages:
   - `pandas==1.5.3`
   - `torch==2.3.0`
   - `transformers==4.51.3`
   - `bitsandbytes==0.45.5`
4. Configure global variables:
   - Path to your local model
   - Model name
   - Output directory
   - Desired hyperparameters
5. Update data import paths with your dataset.
6. Run the **Zero-Shot** and/or **Few-Shot Inference** cells in the notebook.

---

### ☁️ Option 2: Using a Microsoft AI Foundry Model

1. Set up your model in **Microsoft Azure** according to your organization's specifications.
2. Download the notebook: [Single LLM-as-a-Judge (Azure)](02_LLM_as_a_Judge/single_llm_as_a_judge_azure.ipynb).
3. Install the required packages:
   - `openai`
   - `pandas`
   - `numpy`
   - `azure.identity`
4. Configure global variables:
   - API version
   - Azure endpoint
   - Model deployment name
5. Update data file paths.
   - **Note:** Your CSV file should include two columns: `record_id` and `prompt`. Use the [PDSQI Prompt Creation Notebook](02_LLM_as_a_Judge/pdsqi_create_prompt.ipynb) to format your dataset using the PDSQI Evaluation Instrument.
6. Run the **API Calls** cell to generate evaluations.

---

### 🧪 Option 3: Using Epic’s Open Source Script

Use the official script provided by Epic’s open-source initiative:

👉 [Epic Evaluation Instruments – PDSQI-9](https://github.com/epic-open-source/evaluation-instruments/tree/main/instruments/pdsqi_9)

---

## 👨‍⚖️👩‍⚖️ Multiple LLMs-as-Judges Workflow


### Option 1: Using only Microsoft AI Foundry Models

1. Review [AI Foundry Judges Workflow Notebook](02/LLM_as_a_Judge/multiple_llms_as_judges_azure.ipynb) for an overview of implementation.
2. Download the notebook: [Multiple LLMs-as-Judges Example Usage](02_LLM_as_a_Judge/multiple_llms_as_judges_example_usage.ipynb).
3. The following packages will be required for execution:
   - `openai`
   - `pandas`
   - `numpy`
   - `azure.identity`
   - `autogen_agentchat`
   - `autogen_core`
   - `autogen_ext`
   - `tqdm`
4. Adjust example usage notebook for your dataset and model deployment details

---

### Option 2: Using a Microsoft AI Foundry Model and Local Model

1. Review [AI Foundry and Local Judges Workflow Notebook](02/LLM_as_a_Judge/multiple_llms_as_judges_azure+ollama.ipynb) for an overview of implementation.
2. Download the notebook: [Multiple LLMs-as-Judges Example Usage](02_LLM_as_a_Judge/multiple_llms_as_judges_example_usage.ipynb).
3. The following packages will be required for execution:
   - `openai`
   - `pandas`
   - `numpy`
   - `azure.identity`
   - `autogen_agentchat`
   - `autogen_core`
   - `autogen_ext`
   - `tqdm`
4. Adjust example usage notebook for your dataset and model deployment details

---

## 🏋️‍♂️ Training a Customized LLM-as-a-Judge

### 🧑‍🏫 *Supervised Fine-Tuning*

1. Ensure your model is compatible with Hugging Face Transformers and TRL.
2. Download the notebook: [SFT for Single LLM-as-a-Judge](02_LLM_as_a_Judge/pdsqi_llm_as_a_judge_sft.ipynb).
3. Install the required packages:
   - `numpy==1.24.4`
   - `trl==0.18.2`
   - `peft==0.15.2`
   - `datasets==3.6.0`
   - `pandas==1.5.3`
   - `torch==2.3.0`
   - `transformers==4.51.3`
   - `bitsandbytes==0.45.5`
4. Configure global variables:
   - Path to your local model
   - Model name
   - Output directory
   - Desired hyperparameters for QLORA and training
5. Update data import paths with your dataset.
      - **Note:** Use the [PDSQI Prompt Creation Notebook](02_LLM_as_a_Judge/pdsqi_create_prompt.ipynb) to format your prompt using the PDSQI Evaluation Instrument.
6. Run cells

---

### 🎯 *Direct Preference Optimization*

1. Use the model saved following SFT
2. Download the notebook: [DPO for Single LLM-as-a-Judge](02_LLM_as_a_Judge/pdsqi_llm_as_a_judge_dpo.ipynb).
3. Install the required packages:
   - `trl==0.18.2`
   - `peft==0.15.2`
   - `datasets==3.6.0`
   - `pandas==1.5.3`
   - `torch==2.3.0`
   - `transformers==4.51.3`
   - `bitsandbytes==0.45.5`
4. Configure global variables:
   - Path to your local model and SFT model
   - Model name
   - Output directory
   - Desired hyperparameters for QLORA and training
5. Update data import paths with your dataset.
      - **Note:** Use the [PDSQI Prompt Creation Notebook](02_LLM_as_a_Judge/pdsqi_create_prompt.ipynb) to format your prompt using the PDSQI Evaluation Instrument.
6. Run cells