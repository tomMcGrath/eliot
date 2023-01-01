# Dataset loading
I'm going to use the HuggingFace dataset loading library to get raw text. This is easy to use, for example:

```
import datasets

dataset = datasets.load_dataset('wikitext', name='wikitext-2-raw-v1', split='train')
```

will load WikiText2, which I'm going to use for a first run. (Interestingly, WikiText2 splits at the section/subsection level and includes a bunch of blank entries - these choices don't seem ideal). For now I'm just interested in loading data for training a tokeniser, so don't want to get too caught up in data processing - we're only interested in word-level statistics here.