import collections
import math
import tqdm

class CharacterTokeniser(object):
    """Class to process a sequence of characters into tokens with optional batching."""

    def __init__(self, vocab_size, special_chars=['[UNK]', '[PAD]']):
        """Initialise the tokeniser."""
        self.vocab_size = vocab_size
        self.special_chars = special_chars
        self.num_nonspecial_chars = self.vocab_size - len(self.special_chars)
        self.is_trained = False

    def _get_chars_and_counts(self, dataset):
        """Takes a HF dataset and returns a dictionary {char: count}."""
        char_counts = collections.defaultdict(int)
        for record in tqdm.tqdm(dataset):
            for char in record['text']:
                char_counts[char] += 1
                
        return char_counts

    def train(self, dataset):
        """Trains the tokeniser on a HF dataset."""
        char_counts = self._get_chars_and_counts(dataset)
        chars_by_count = [(char, char_counts[char]) for char in char_counts]
        chars_by_count = sorted(chars_by_count, key=lambda x:x[1], reverse=True)
        chars_to_keep = [(char, 0) for char in self.special_chars] + chars_by_count[:self.num_nonspecial_chars]

        # Setup character <-> token mapping
        char_to_tok = {}
        tok_to_char = {}
        for i, (char, _) in enumerate(chars_to_keep):
            char_to_tok[char] = i
            tok_to_char[i] = char

        # Store results
        self._char_to_tok = char_to_tok
        self._tok_to_char = tok_to_char
        self.is_trained = True

    def tokenise(self, sequence):
        """Tokenise data using a learned tokeniser."""
        if not self.is_trained:
            raise ValueError('Tokeniser must be trained first')
        tokens = []
        seq_len = 0
        
        # Tokenise
        for char in sequence:
            if char in self._char_to_tok:
                tok = self._char_to_tok[char]
            else:
                tok = self._char_to_tok['[UNK]']
                
            tokens.append(tok)
            seq_len += 1
        return tokens

    def pad(self, tokens, target_batch_size):
        """Pad a batch of tokens to an intended batch size."""
        seq_len = len(tokens)
        num_to_pad = target_batch_size - seq_len
        pad_tok = self._char_to_tok['[PAD]']
        tokens += [pad_tok] * num_to_pad
        return tokens

    def tokenise_and_batch(self, sequence, batch_size):
        """Tokenise a sequence and split it into batches with padding."""
        tokens = self.tokenise(sequence)
        seq_len_in_tokens = len(tokens)
        split_count = math.ceil(seq_len_in_tokens / batch_size)
        splits = [tokens[i*batch_size:(i+1)*batch_size] for i in range(split_count)]
        splits[-1] = self.pad(tokens, batch_size)
        return splits

    def detokenise(self, tokenised_data):
        """Detokenise data that uses this tokeniser's char <-> token map."""
        return [self._tok_to_char[tok] for tok in tokenised_data]

    def detokenise_to_string(self, tokenised_data):
        """Convenience method for detokenising to a string rather than a list."""
        return ''.join(self.detokenise(tokenised_data))