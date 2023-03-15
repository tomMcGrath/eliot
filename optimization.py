import torch
import numpy as np


def build_lr_fn(warmup_steps, lr_max, cosine_steps):
  """Builds a lr schedule function with linear warmup and cosine decay."""
  def lr_fn(t):
    if t <= warmup_steps:  # linear warmup
      return t * lr_max / warmup_steps

    else:  # cosine decay -> 0
      t_cos = t - warmup_steps
      return 0.5 * lr_max * (1 +  np.cos(np.pi * t_cos / cosine_steps))

  return lr_fn


def build_optimizer_and_scheduler(model, training_config):
  """Creates an AdamW optimizer and warmup + cosine decay lr scheduler."""
  optimizer = torch.optim.AdamW(model.parameters(),
                                lr=1,  # let the lr_fn handle the learning rate
                                betas=(training_config.adam_beta1,
                                       training_config.adam_beta2),
                                weight_decay=training_config.weight_decay)

  lr_fn = build_lr_fn(warmup_steps=training_config.warmup_steps,
                      lr_max=training_config.lr_max,
                      cosine_steps=training_config.post_warmup_steps)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
  return optimizer, scheduler
