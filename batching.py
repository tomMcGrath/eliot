import collections
import numpy as np
import torch


def make_processor(encoding_fn, seq_len):
    def process_fn(txt):
        tokens = torch.tensor(encoding_fn(txt))
        targets = tokens.roll(-1)
        tokens = tokens[:-1]  # nothing to predict at last token
        targets = targets[:-1]  # first token has rolled around
        
        split_tokens = tokens.split(seq_len)
        split_targets = targets.split(seq_len)
        
        if len(split_tokens[-1]) < seq_len:
            split_tokens = split_tokens[:-1]
            split_targets = split_targets[:-1]
            
        return [{'tokens': split_tokens[i], 'targets': split_targets[i]} 
                for i in range(len(split_tokens))]
    return process_fn


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
            next_datum = self.get_next()

        if not self._data_deque:  # prefer refilling before IndexError
            self._refill()

        return next_datum


class BatchedDataSource:

    def __init__(self, data_sources):
        self._data_sources = data_sources

    def get_next(self):
        """Collates the outputs of individual data sources and batches."""
        unbatched = [ds.get_next() for ds in self._data_sources]

        # This process feels slow but I haven't profiled yet
        batched = {}
        for k in unbatched[0].keys():  # assume all keys the same
            batched[k] = torch.stack([x[k] for x in unbatched])

        return batched
