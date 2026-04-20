# Overview
The ProcessInputFiles python class requires as input the following CSV files:

* note_file -> Expected Format: Dataframe with 2 columns 'record_id' (int) and 'notes' (str)
* summary_file -> Expected Format: Dataframe with 3 columns 'record_id' (int), 'summary' (str), and 'target_specialty' (str)
* score_file -> Expected Format: Dataframe with 1 columns 'record_id' (int) filtered to those in test set
* train_file -> Expected Format: Dataframe with 2 columns 'record_id' (int) and 'scores' (str) from training set

Returns a dataframe with columns "record_id" and "input".

If write_to_disk==True, then it will also write the files to disk at the specified directory location with the filename "pdsqi_input_to_llm_as_a_judge_few_shot.csv" or "pdsqi_input_to_llm_as_a_judge_zero_shot.csv"
* Note: this is only applicable if you are using the `create_zeroshot_csv()` or `create_fewshot_csv()` class methods.

Important: the code assumes that the train dataset does not have overlap or bleed-over of data into the test dataset.

If running zeroshot or few shot, then you need to also provide a human-readable `model_name` variable, and the exact path to the LLM as the `model_id` variable. You may wish to customize the LLM parameter dataclass in the class initialization.

# Usage with example for creating zeroshot or fewshot datasets

    # initialize object
    inputs = ProcessInputFiles(note_file = "df_noteFile.csv", summary_file = "df_summFile.csv", score_file = "df_scoreFile.csv", train_file = "df_trainFile.csv")

    # create rubric text and required pandas dataframes
    rubric_text = inputs.rubric_text()
    note_df = inputs.createNoteFile_df(inputs.note_file)
    summ_df = inputs.createSummaryFile_df(inputs.summary_file)
    score_df = inputs.createScoreFile_df(inputs.score_file)
    train_df = inputs.createTrainingData_df(inputs.train_file)

    # create zeroshot dataset
    input_df = inputs.create_zeroshot_csv(score_df, summ_df, note_df, output_file_directory="output_tmp", write_to_disk=True)
    # create few shot dataset
    input_df = inputs.create_fewshot_csv(train_df=train_df, score_df=score_df, summary_df=summ_df, note_df=note_df, num_shots=5, output_file_directory="output_tmp", write_to_disk=True)

# Usage with example for running zeroshot or fewshot

    # initialize object
    inputs = ProcessInputFiles(note_file = "df_noteFile.csv", summary_file = "df_summFile.csv", score_file = "df_scoreFile.csv", train_file = "df_trainFile.csv")

    # create rubric text and required pandas dataframes
    rubric_text = inputs.rubric_text()
    note_df = inputs.createNoteFile_df(inputs.note_file)
    summ_df = inputs.createSummaryFile_df(inputs.summary_file)
    score_df = inputs.createScoreFile_df(inputs.score_file)
    train_df = inputs.createTrainingData_df(inputs.train_file)

    # OPTIONAL: modify the model.generate parameters
    inputs.params.temperature = 0.70

    # run zeroshot workflow
    inputs.run_zeroshot(score_df, summ_df, note_df, model_id="llama3.1-8B", model_name="llama", output_file_directory="output_tmp")
    # create few shot dataset
    inputs.run_fewshot(score_df, summ_df, note_df, train_df, model_id="llama3.1-8B", model_name="llama", num_shots=5, output_file_directory="output_tmp")


# Example Toy Datasets

training.csv

| record_id | scores |
| --- | --- |
| 1 | high |
| 2 | low |
| 3 | very high |
| 4 | very low |
| 5 | high |

summary.csv

| record_id | summary | target_specialty |
| --- | --- | --- |
| 1 | Patient is alive | Internal Medicine |
| 2 | Patient is dead | Critical Care |
| 3 | Patient is alive | Pediatrics |
| 4 | Patient is mostly dead | OBGYN |
| 5 | Patient is alive | Internal Medicine |

scores.csv

| record_id | notes |
| --- | --- |
| 1 | Note1 |
| 2 | Note2 |
| 3 | Note3 |
| 4 | Note4 |
| 5 | Note5 |

notes.csv

| record_id | notes |
| --- | --- |
| 1 | Note1 |
| 2 | Note2 |
| 3 | Note3 |
| 4 | Note4 |
| 5 | Note1 |