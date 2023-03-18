import batching
import config
import data_loading
import models
import numpy as np
import optimization
import sys
import tiktoken
import torch
import wandb
import wikitext_utils


# Load config
config_path = sys.argv[1]
with open(config_path, 'r') as jsonfile:
  config_json = jsonfile.read()
cfg = config.Config.from_json_str(config_json)

print('Running prelaunch checks')
prelaunch_check_results = cfg.passes_prelaunch_checks()
if prelaunch_check_results.passes:
  print('Prelaunch checks passed, continuing with setup')
else:
  prelaunch_errs = prelaunch_check_results.errs
  raise ValueError(f'Prelaunch checks failed with errors:\n{prelaunch_errs}')

# Setup logging
wandb.init(project='eliot-c4')

# Load data
dataset, tokeniser = data_loading.build_from_config(cfg.dataset_config)

# Create model
# TODO: try rounding vocab_size to nearest multiple of 64
model_config = cfg.model_config

print('Initialising model')
model = models.TransformerModel(
    model_config.vocab_size,
    model_config.num_layers,
    model_config.d_model,
    model_config.n_heads,
    model_config.d_head,
    model_config.max_len,
    model_config.d_mlp,
)
print('Model built')

print('Setting up training')
training_config = cfg.training_config

optimizer, scheduler = optimization.build_optimizer_and_scheduler(
  model, training_config)
loss_fn = torch.nn.CrossEntropyLoss()

cuda0 = torch.device('cuda:0')
model = model.to(cuda0)
wandb.watch(model)
print('Training setup complete')

print('Setup complete! Beginning training')
for step in range(training_config.total_num_steps):
  # Forward pass
  batched_data = dataset.get_next_batch()
  # TODO: benchmark vs with pin_memory() and non_blocking=True
  tokens = batched_data['tokens'].to(cuda0)
  targets = batched_data['targets'].to(cuda0)
  logits = model(tokens)
  loss = loss_fn(logits.transpose(1, 2), targets)

  # Backprop
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  scheduler.step()

  # Logging
  train_loss = loss.cpu().detach().numpy().item()
  last_lr = scheduler.get_last_lr()[0]
  wandb.log(
    {'train_loss': train_loss,
     'lr': last_lr,
     })