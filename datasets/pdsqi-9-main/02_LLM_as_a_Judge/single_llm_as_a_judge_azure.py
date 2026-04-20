from azure.identity import DefaultAzureCredential, AzureCliCredential, InteractiveBrowserCredential, get_bearer_token_provider
from openai import AzureOpenAI, BadRequestError
import os
import openai
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import time
from pathlib import Path
import requests
import urllib
import glob
import re
import random

# set input and output data
# key is the input test filename
# # value0 is the JSON output
# # value1 is the predicted text CSV output

fileDict = [
     {"input_file.csv" : [
         "run0_raw_json.jsonl",
         "run0_dataframe.csv"
     ]},
     {"input_file.csv" : [
         "run1_raw_json.jsonl",
         "run1_dataframe.csv"
     ]}
]

# setup Azure environment

api_version = "INSERT API VERSION"
endpoint = "INSERT AZURE ENDPOINT"
deployment = "DEPLOYMENT NAME"

token_provider = get_bearer_token_provider(
    AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
)

# functions
def chatGPT_API(client, prompt, d_name):

    """
    Calls an Azure OpenAI chat completion API using the provided client and user prompt.

    Parameters:
        client (openai.Client): The Azure OpenAI client instance.
        prompt (str): The user's input prompt to send to the model.

    Returns:
        tuple:
            - r_json (dict): The raw JSON response (or error info if an exception occurred).
            - response (str): The generated content from the assistant, or an empty string if an error occurred.
    """

    deployment_name = d_name 
    response = "NONE"
    r_json = ""

    # modify this section as needed for user role, system role, and settings
    try:
        response = client.chat.completions.create(
            model = deployment_name,
            messages = [
                {
                    # Do not include for DeepSeek Models instead append to user prompt
                    "role" : "system",
                    "content" : "You are a summarization quality expert that specializes in text analysis and reasoning. Please start your response with '<think>' at the beginning. Provide your reasoning when generating the final output."
                },
                {
                    "role" : "user",
                    "content" : prompt
                }
            ],
            # temperature = 0.01,
            # top_p = 0.95,
            # max_tokens = 400,
        )
    except BadRequestError as e:
        print(e)
        r_json = {'error': {'message': "The response was filtered due to the prompt triggering Azure OpenAI's content management policy."}}
    try:
        r_json = response.to_json()
        r_json = json.loads(r_json)
        response = r_json["choices"][0]["message"]["content"]
    except KeyError:
        print("Error, no content key in JSONL response")
        print(s)
        print("\n")
        print("###")
        response = ""
        print(r_json)
        print("\n")
    return r_json, response

def runQueries(l_tuple, archive_output_filename, output_file_name, client=client, deployment=deployment):

    """
    Executes a batch of prompt-response queries using a chat model and logs the outputs.

    Parameters:
        l_tuple (list of tuples): Each tuple must be in the form (input_text, record_id).
        archive_output_filename (str): Path to the file where raw JSON responses will be archived.
        output_file_name (str): Path to the CSV file where the summarized output will be saved.
        client (openai.Client): Azure OpenAI client to use for querying the chat API (default: `client`).

    Outputs:
        - A JSON archive file (`archive_output_filename`) containing a list of raw response dictionaries from the model.
        - A CSV file (`output_file_name`) with the following columns:
            - "PredictedText": The model's generated text.
            - "Record_ID": The original ground truth text provided.
            - "InputText": The input prompt sent to the model.
    """

    l_results = []
    ct = 0
    for idx in tqdm(range(len(l_tuple))):
        i = l_tuple[idx]
        input_text = i[0]
        gt_text = i[1]
        j,s = chatGPT_API(client, input_text, deployment)
        l_results.append((j, s, gt_text, input_text))
        loopCt = None
        try:
            loopCt = j["usage"]["total_tokens"]
        except KeyError:
            print("Warning, unable to determine token count for loop")
        try:
            loopCt = int(loopCt)
        except TypeError:
            print("Warning, loopCt value was NoneType")
            loopCt = 0
        except ValueError:
            print("Warning, unable to cast json key 'total_tokens' as type int")
            loopCt = 0
        ct += loopCt
        # optional: uncomment to add in a stop condition
        # if ct > 1000000:
        #    print("total tokens exceeded 1000000, pausing for 30 seconds")
        #    ct = 0
        #    time.sleep(30)

    archive_output = [x[0] for x in l_results]
    with open(archive_output_filename, 'w') as fWrite:
        fWrite.write("[\n")
        lastCount = len(archive_output) - 1
        ct = 0
        for i in archive_output:
            s = str(i)
            if ct < lastCount:
                s += ",\n"
            fWrite.write(s)
            ct += 1
        fWrite.write("\n]")

    df_output = pd.DataFrame({
        "PredictedText" : [x[1] for x in l_results],
        "Record_ID" : [x[2] for x in l_results],
        "InputText" : [x[3] for x in l_results]
    })
    df_output.to_csv(output_file_name, index=False)
    print(f"wrote file to disk: {output_file_name}")

# run workflow
def run_workflow(client: AzureOpenAI, deployment: str, fileDict: list[dict[str, [list[str]]]]):
    print(f"Using deployment {deployment}")
    for itm in fileDict:
        for k,v in itm.items():
            print(f"Running workflow for {k}")
            print(f"Output JSON file will be named {v[0]}")
            inFileName = str(k)
            outFileName1 = v[1]
            archOutFile1 = v[0]
            if Path(outFileName1).exists():
                print(f"WARNING! Found existing file with name {outFileName1}\nthis will be overwritten if you proceed!")
            if Path(archOutFile1).exists():
                print(f"WARNING! Found existing file with name {archOutFile1}\nthis will be overwritten if you proceed!")
            df = pd.read_csv(str(k), header=None, names=["prompt"]) # this may need to be changed
            df["record_id"] = [x for x in range(len(df))]
            df = df.copy()
            l_inputs = df["prompt"].tolist()
            l_gts = df["record_id"].tolist()
            l1 = list(zip(l_inputs, l_gts))
            runQueries(l1, archOutFile1, outFileName1, client=client, deployment=deployment)
