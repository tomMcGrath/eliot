# Batching notes
Now we have a working tokeniser, the next question is how to arrange its outputs into batches for training. This could have a big impact on training performance by making batches very autocorrelated if we do it wrong. On the other hand, there's the possibility of making this arbitrarily complicated for potentially ~zero performance gain. We want to have a sensible baseline to compare with.

## Batching strategies
This isn't something I've seen written about very much, so I'm going to make up some hopefully-useful terminology. Let's say we have two documents, `doc1` and `doc2` that we want to tokenise and batch, a batch size of two sequences, and a sequences length of 3.
```python
doc1 = 'abcdef'
doc2 = 'ghijk'
```

### Sequential documents
One obvious approach is to go sequentially, filling the entire batch as we go. This would lead to the following outcome (ignoring tokenisation for now):
```python
batch1 = [['a', 'b', 'c'], ['d', 'e', 'f']]
batch2 = [['g', 'h', 'i'], ['j'. 'k', '[PAD]']]
```
There's clearly a lot of autocorrelation here: batch 1 is only from document 1 and batch 2 is only from document 2. With larger batch sizes and relatively short documents this problem will be much less pronounced, but it still seems somewhat undesirable.

### Parallel documents
An alternative approach is to assign a stream of documents to each batch element separately. In the case, the first element in each batch will be drawn from document 1, and the second element will be drawn from document 2, which will repeat in each batch.
```python
batch1 = [['a', 'b', 'c'], ['g', 'h', 'i']]
batch2 = [['d', 'e', 'f'], ['j', 'k', '[PAD]']]
```
In this case we've got much less autocorrelation. 

## Problems with padding
There's a potential problem, however: this approach can lead to considerably more padding. Say that we still had batch size 2, but that the length of each sequence was now 4. In this case, parallel documents yields the following batches:
```python
batch1 = [['a', 'b', 'c', 'd'], ['g', 'h', 'i', 'j']]
batch2 = [['e', 'f', '[PAD]', '[PAD]'], ['k', '[PAD]', '[PAD]', '[PAD]']]
```
There's mostly padding in the final batch! Of course this is a worst-case situation but when sequence lengths vary (and especially when the sequence length is near the document length) this could slow down learning a lot.

### Approaches to resolving these problems
Three main approaches to padding issues immediately suggest themselves:
1. Ignore them!
2. Drop the final sequence from each document (the one that would contain padding).
3. Pack sequences - when a document ends, don't pad the sequence but instead put a special `'[EOF]'` character and start the next document.

### Advantages and disadvantages of these approaches
Option 1 has the huge advantage of simplicity but not much else to commend it. It's so simple that I probably won't learn much from it. Option 2 is interesting and probably worthwhile. Option 3 will introduce more complexity into the pipeline (e.g. attention masking across sequences), but is probably relatively easy to generalise from option 2. I think I'll start with option 2.