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

## Code & personal organisation
This is the first time for a while that I've been able to do completely greenfield development and to be honest the freedom is a little bit intimidating - what if I get something wrong? What if my code is poorly organised? How do I even do a whole load of stuff I've taken for granted in the Google ecosystem (e.g. continuous integration, Python environments)? I think the answer is to not worry too much about this; this is a private project and I can make mistakes if I want to! In fact, that's part of the point. To be honest, I can probably code myself out of any problems I can code myself into - the key is just to write a lot of code!

I'm particularly concerned about architecting this correctly, and am getting a bit paralysed with this. This is complicated by the fact that I'm not used to most of the tools I'm working with here (e.g. HuggingFace library, PyTorch) and so need to write a lot of sem-throwaway code. I think the way forward is probably to use notebooks to get used to these new tools and do prototyping.