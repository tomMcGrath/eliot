import collections
import char_tokeniser


def make_processor(tokeniser, subseq_len, discard_last=False):
    def process_sequence(seq):
        """Process a sequence into batches of tokens and targets."""
        if not seq:
            return None

        tokens = tokeniser.tokenise(seq)
        targets = [tokens[i+1] for i in range(len(tokens) - 1)]
        split_tokens = char_tokeniser.split_into_subseqs(tokens[:-1], subseq_len)
        split_targets = char_tokeniser.split_into_subseqs(targets, subseq_len)

        # Pad or discard
        if discard_last:
            split_tokens = split_tokens[:-1]
            split_targets = split_targets[:-1]
        else:
            split_tokens[-1] = tokeniser.pad(split_tokens[-1], subseq_len)
            split_targets[-1] = tokeniser.pad(split_targets[-1], subseq_len)

        data = [{'tokens': split_tokens[i], 'targets': split_targets[i]} for i in range(len(split_tokens))]
        return data
    return process_sequence


class DataSource:
    
    def __init__(self, data_iterator, process_fn):
        """Initialise the DataSource."""
        self.data_iterator = data_iterator
        self.process_fn = process_fn
        self._data_deque = collections.deque()
        
    def _refill(self):
        """Gets a new sequence from the data iterator and processes for the model."""
        processed_next_record = None
        while processed_next_record is None:
            next_record = next(self.data_iterator)
            try:
                processed_next_record = self.process_fn(next_record)
            except ValueError:
                continue
        self._data_deque.extend(processed_next_record)
    
    def get_next(self):
        """Get next datum and refill if necessary."""
        try:
            next_datum = self._data_deque.popleft()
        except IndexError:
            self._refill()
            next_datum = self._data_deque.popleft()
            
        if not self._data_deque:  # prefer refilling before IndexError
            self._refill()
            
        return next_datum