import collections

class DataSource:
    
    def __init__(self, data_iterator, tokeniser, seq_len):
        """Initialise the DataSource."""
        self.data_iterator = data_iterator
        self.tokeniser = tokeniser
        self.seq_len = seq_len
        self._data_deque = collections.deque()
        
    def _refill(self):
        """Gets a new sequence from the data iterator and tokenise."""
        tokenised_next_record = None
        while tokenised_next_record is None:
            next_record = next(self.data_iterator)
            try:
                tokenised_next_record = self.tokeniser.tokenise_and_batch(
                    next_record['text'][0], self.seq_len)
            except ValueError:
                continue
        self._data_deque.extend(tokenised_next_record)
    
    def get_next(self):
        """Get next datum and refill if necessary."""
        try:
            next_datum = self._data_deque.popleft()
        except IndexError:
            self._refill()
            
        if not self._data_deque:  # prefer refilling before IndexError
            self._refill()