# Usage

Modify the `fileDict` variable to have the input CSV filename and the output filenames as follows:
* the dictionary key must be the filename, in CSV format, with an input column that contains the full prompt that will be sent to the LLM in an API call, and an output column that contains the correct gold truth answer
* the dictionary value at index0 is the full JSON output including the metadata
* the dictionary value at index1 is the CSV file that includes the predicted text

Example:

    fileDict = [
        {"input_file.csv" : ["run0_json.jsonl", "run0_dataframe.csv"]
    ]

Where input_file.csv looks something like this:

| input | output |
| --- | --- |
| As a medical professional, please examine the medical note and offer a prediction on the patient's outcome. Patient was admitted to the ED with 104.1F fever, 150/110 BP, 150 HR. | patient is alive |

Next, provide the deployment name, the API version, and the Azure endpoint, all of which can be found in AI Foundry for the deployment you have created.
* Note1: The API version may not be necessary, because Azure has recently changed its API requirements
* Note2: The code assumes you are using Microsoft's EntraID Authentication with AD. The usage with API keys will be slightly different, but usually follows the example code in AI Foundry closely.

Finally, run the python script via `python single_llm_as_a_judge_azure.py`
