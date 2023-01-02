# Tokenisation notes
Past character- and word-level tokenisation these notes on tokenisers are based largely on the excellent notes from Hugging Face (HF). I'm going to start with character- and word-level tokenisation as these are really simple to implement and will let me do simple training runs. After this I'll try more complex tokenisers, starting with WordPiece tokenisation. First I'll make an inefficient single-thread pure Python implementation, then worry about efficiency.

## Character-level tokenisation
Character-level tokenisation is simple - we just tokenise each character individually. There are ~5k distinct characters in WikiText-103 and ~500B total characters, with the top ~100 characters making up the vast majority of character occurrences.

### How do we implement character-level tokenisation?
Implementing a character-level tokeniser should be pretty simple:

*Training*
1. Scan the corpus to get the complete set of characters and their counts.
2. Sort by size and truncate at a fixed size.
3. Add special characters (`[PAD]`, `[UNK]`, `[BOS]`/`[EOS]`).

*Tokenisation*
1. Scan through the data to tokenise, replacing not-in-vocab chars with `[UNK]`.
2. Pad the tokenised data with the `[PAD]` token to hit the intended sequence length.

A possible improvement that might help with meta-learning would be to have `[UNK1]`, `[UNK2]`, ..., `[UNKN]` tokens for each unknown character that's encountered in the document. This could help with meta-learning.

### What is good and bad about character-level tokenisation?
An extremely good thing about character-level tokenisation is that it's very simple to implement! There's no real choices apart from setting the vocab size - even word-level tokenisation has more choices to make: for instance, how do we handle punctuation? This makes character-level tokenisation a great choice for implementing a first tokeniser so we can try out the interface in the context of a full training run.

A possible downside of character-level tokenisation is that I don't think that embeddings make particularly much sense at the character level because there's no real semantics to capture here - what are the semantics of the letter "a", for instance? Perhaps it's fine - the product of the embedding and unembedding matrices can also be seen as simply capturing 1-gram statistics so we can at least get letter co-occurrences. Another major thing that concerns me is the potential for the model to spend a lot of capacity on simply learning which words exist.