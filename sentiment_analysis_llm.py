from openai import OpenAI
import json
from tqdm import tqdm
import pandas as pd
import math
import os
import matplotlib.pyplot as plt

# Load the Sentiment140 dataset
columns = ["target", "id", "date", "flag", "user", "text"]
data = pd.read_csv("sentiment140/versions/2/training.1600000.processed.noemoticon.csv",
                   encoding='latin-1', names=columns)
# target=0=negative and target=4=positive
tweeter_data = data[["target", "text"]]
tweeter_data["target"] = tweeter_data["target"].map({0: "negative", 4: "positive"})

# Initialize the OpenAI client
client = OpenAI(api_key='add your api key')

# List to store the results
batch = []
batch_size = 50000  # Set max lines per file

# Loop through the tweets and prepare prompts
for ind in tqdm(tweeter_data.index):
    tweet = tweeter_data.iloc[ind, 1]

    prompt = (
        "Your role is to analyze the sentiment of sentences and classify it as Positive or Negative. Provide a "
        "justification for your classification based solely on the content of the text. "
        "Example 1: "
        "Text: \"I'm thrilled with the new update; it works flawlessly!\" "
        "Sentiment: Positive "
        "Justification: The speaker expresses happiness and approval with words like \"thrilled\" and \"flawlessly.\" "

        "Example 2: "
        "Text: \"I'm unhappy with the customer service I received today.\" "
        "Sentiment: Negative "
        "Justification: The speaker expresses dissatisfaction with the phrase \"unhappy with the customer service.\" "
        f"Now, analyze the following text: {tweet}"
    )

    # Create a batch entry
    new_batch = {"custom_id": f"request-{ind}",
                 "method": "POST",
                 "url": "/v1/chat/completions",
                 "body": {"model": "gpt-4o",
                          "messages": [{"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                                       {"role": "user", "content": prompt}],
                          "max_tokens": 200,
                          "response_format": {"type": "json_object"},
                          "temperature": 0}
                 }

    batch.append(new_batch)

    # Save and clear the batch if it reaches the batch size limit
    if len(batch) == batch_size or ind == tweeter_data.index[-1]:  # Save every batch_size or at the last tweet
        file_num = math.ceil(ind / batch_size)
        file_path = f'batch_files/sentiment_good_prompt_{file_num}.jsonl'
        with open(file_path, 'w') as file:
            for entry in batch:
                json_line = json.dumps(entry)
                file.write(json_line + '\n')
        batch = []  # Clear batch after saving

# Loop through each file in the directory and submit it
for file_name in os.listdir('batch_files'):
    if file_name.startswith("sentiment_good_prompt_"):
        file_path = os.path.join('batch_files', file_name)

        # Upload the batch file
        with open(file_path, "rb") as file:
            batch_input_file = client.files.create(file=file, purpose="batch")

        # Get the uploaded file's ID
        batch_input_file_id = batch_input_file.id

        # Run the batch online
        client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        print(f"Batch for {file_name} has been submitted successfully.")


"""
Results
"""
sentiment_llm = pd.read_csv('results.csv')

# Check accuracy of predictions
sentiment_llm["is_correct"] = sentiment_llm["Sentiment"] == sentiment_llm["target"]
accuracy_counts = sentiment_llm["is_correct"].value_counts(normalize=True) * 100

# Plot the comparison of correct vs incorrect predictions
plt.figure(figsize=(10, 8))
accuracy_counts.plot(kind="bar", color=["green", "red"], legend=False)
plt.title("Accuracy of Sentiment Predictions")
plt.xlabel("Prediction Accuracy")
plt.ylabel("Frequency")
plt.xticks([0, 1], ["Correct", "Incorrect"], rotation=0)
plt.show()




