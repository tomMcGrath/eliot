from datasets import load_dataset
from datetime import datetime
import tiktoken
import wandb


# Set run parameters
min_doc_len = 1024
target_num_tokens = int(1e8)
output_base_path = 'filtered_c4/'
burnin_time_in_seconds = 10

# Use W&B to log progress for long runs
wandb.init(project='eliot-data')

# Setup streaming dataset
dataset = load_dataset("c4", "en", split="train", streaming=True)

# Use gpt2 encoding to determine number of tokens
encoding = tiktoken.get_encoding('gpt2')

# Stream the dataset and get the docs that are min_doc_len or longer
num_docs_so_far = 0
num_toks_so_far = 0
start_time = datetime.now()
for num_docs_checked, example in enumerate(dataset):
    doc_len = len(encoding.encode(example['text']))

    if doc_len >= min_doc_len:
        # Write the file
        with open(output_base_path + f'{num_docs_so_far}.txt', 'w+') as f:
            f.write(example['text'])

        # Update the output
        num_docs_so_far += 1
        num_toks_so_far += doc_len

        # Get tokens/sec and time remaining
        time_elapsed = (datetime.now() - start_time).seconds
        time_elapsed = max(time_elapsed, 1)  # prevent divide by 0
        tokens_per_sec = num_toks_so_far / time_elapsed
        num_tokens_remaining = target_num_tokens - num_toks_so_far
        seconds_remaining = num_tokens_remaining / tokens_per_sec 

        # Log progress when past burnin point
        if time_elapsed > burnin_time_in_seconds:
            wandb.log({'docs_written': num_docs_so_far,
                       'docs_seen': num_docs_checked,
                       'tokens': num_toks_so_far,
                       'tokens_per_sec': tokens_per_sec,
                       'num_tokens_remaining': num_tokens_remaining,
                       'seconds_remaining': seconds_remaining})

    # Stop if we've got enough data
    if num_toks_so_far >= target_num_tokens:
        break
