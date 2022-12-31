# What's this about?
This project is intended as a learning experience to improve my engineering skills and familiarise me with the details of the language modelling stack. I want to write a language model training pipeline that I can run both on my machine (2x3090 Ti) and on a larger machine (hopefully enough to scale to a 1B param training run). 

# Important decisions
## How much is going to be written from scratch?
This is something I expect will fluctuate quite a lot. It's tempting to sub in a lot of components like Hugging Face tokenisers and dataset loaders, but doing so would lose a lot of the learning experience. These components are likely to be faster/more general than what I'll write so it seems like it'll be tempting to leave them in rather than decrease performance to give myself a learning opportunity. On the other hand, I don't want to implement everything from the bottom up, mintorch-style. I think a good compromise is to stick with 'standard pytorch' plus:
- Dataset loading (e.g. Hugging Face, torchtext)
- Monitoring (e.g. Weights and Biases)
- Training from PyTorch Lightning (maybe!)

I definitely want to write the following components:
- Tokeniser & post-tokenisation data management
- Model
- Training 
- Sampling
- Evaluation

## What datasets am I targetting?
Possibilities:
- Wikitext
- Shakespeare
- Code
- Common crawl (C4) or the Pile
- Generic text (e.g. finetuning from a folder of text)