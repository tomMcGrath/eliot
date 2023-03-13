import batching
import models
import numpy as np
import tiktoken
import torch
import wandb
import wikitext_utils

# Setup logging
wandb.init(project='eliot')

# Load data
print('Loading data')
dataset = wikitext_utils.load_wikitext_train()
text_iterator = wikitext_utils.make_shuffled_text_iterator(dataset)
encoding = tiktoken.get_encoding('gpt2')
print('Data loaded')

# Create datasource
print('Building datasources')
seq_len = 128
batch_size = 32
process_fn = batching.make_processor(encoding.encode, seq_len)
datasources = [batching.DataSource(text_iterator, process_fn)
                       for _ in range(batch_size)]
batched_data_source = batching.BatchedDataSource(datasources)
print('Datasources built')

# Create model
# TODO: try rounding vocab_size to nearest multiple of 64
vocab_size = encoding.n_vocab
num_layers = 12
d_model = 768
n_heads = 12
d_head = int(d_model / n_heads)
d_mlp = int(4 * d_model)

print('Initialising model')
model = models.TransformerModel(
    vocab_size,
    num_layers,
    d_model,
    n_heads,
    d_head,
    seq_len,
    d_mlp,
)
print('Model built')

# Create optimiser
# Use this inside a LambdaLR
def build_scheduler(warmup_steps, lr_max, cosine_steps):
  def scheduler(t):
    if t <= warmup_steps:  # linear warmup
      return t * lr_max / warmup_steps

    else:  # cosine decay -> 0
      t_cos = t - warmup_steps
      return 0.5 * lr_max * (1 +  np.cos(np.pi * t_cos / cosine_steps))

  return scheduler

lr_max = 6e-4
weight_decay = 1e-1
warmup_steps = int(2e4)
n_steps = int(6e5)
beta1 = 0.9
beta2 = 0.95

optimizer = torch.optim.AdamW(
  model.parameters(), lr=1, betas=(beta1, beta2), weight_decay=weight_decay)
lr_fn = build_scheduler(
  warmup_steps=warmup_steps, lr_max=lr_max, cosine_steps=n_steps-warmup_steps)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
loss_fn = torch.nn.CrossEntropyLoss()

cuda0 = torch.device('cuda:0')
model = model.to(cuda0)
wandb.watch(model)

for step in range(n_steps):
  # Forward pass
  batched_data = batched_data_source.get_next()
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