# Model notes
The aim for this project is to write and train a language model. The main target model is a small-medium autoregressive transformer-based language model.

## Things I'd like to be able to do
1. Next token prediction (obviously).
2. Activation readout (presumably via PyTorch hooks?). More ambitiously, intervene.
3. Compute low-level scaling laws.
4. Small-scale architectural comparisons, e.g. [Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/abs/2212.14034), [How to Train BERT with an Academic Budget](https://arxiv.org/abs/2104.07705).
5. Benchmark evaluation.

## Possible extensions
I'm making notes here of things partly in order that I can come back to them later if I want, but also so I don't have them hanging around taking up space in my brain while I'm trying to do the main thing! A few cool extensions could be:
1. Masked language modelling (e.g. BERT).
2. [Structured state space sequence](https://arxiv.org/abs/2111.00396) (S4) models.

## How can I get there?
As before, the answer is going to be: start with the absolute simplest possible thing and work up from there. I want to get a complete pipeline running first to make sure the design makes sense and avoid breakages. I probably want to develop this as follows:
1. Embedding matrix only - just do bigram modelling!
2. Single-layer RNN with embeddings (optional, use PyTorch implementation).
3. Transformer (implemented myself).

Fortunately the embedding matrix is something I can easily reuse.