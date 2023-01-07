import collections
import char_tokeniser


def process_sequence(tokeniser, seq, subseq_len):
    """Process a sequence into batches of tokens and targets."""
    if not seq:
        return None
    
    tokens = tokeniser.tokenise(seq)
    targets = [tokens[i+1] for i in range(len(tokens) - 1)]
    split_tokens = char_tokeniser.split_into_subseqs(tokens, subseq_len)
    split_targets = char_tokeniser.split_into_subseqs(targets, subseq_len)
    data = [{'tokens': split_tokens[i], 'targets': split_targets[i]} for i in range(len(split_tokens))]
    return data

class DataSource:
    
    def __init__(self, data_iterator, process_fn, seq_len):
        """Initialise the DataSource."""
        self.data_iterator = data_iterator
        self.process_fn = process_fn
        self.seq_len = seq_len
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
            
        if not self._data_deque:  # prefer refilling before IndexError
            self._refill()