import glob
import tiktoken
import torch


def build_from_config(dataset_config):
    encoding = tiktoken.get_encoding(dataset_config.tokenizer)
    process_fn = make_processor(encoding.encode, dataset_config.seq_len)
    dataset = Dataset(data_dir=dataset_config.train_dataset,
                      process_fn=process_fn,
                      batch_size=dataset_config.batch_size)
    return dataset, encoding


def load_doc_and_process(docpath, process_fn):
    with open(docpath, 'r') as f:
        file_contents = f.read()
    return process_fn(file_contents)


def get_file_num(filename):
    return int(filename.split('.txt')[0])


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


class Dataset:
    
    def __init__(self, data_dir, process_fn, batch_size):
        # Store attributes
        self.data_dir = data_dir
        self.process_fn = process_fn
        self.batch_size = batch_size
        
        # Setup data files and fill initial buffer
        data_files = glob.glob('*.txt', root_dir=data_dir)
        self.data_files = sorted(
            data_files, key=lambda x: get_file_num(x), reverse=True)
        self.tokenised_data = []
        self.fill_buffer()
        
    def fill_buffer(self):
        while len(self.tokenised_data) < self.batch_size:
            next_doc_data = load_doc_and_process(
                f'{self.data_dir}/{self.data_files.pop()}', self.process_fn)
            if next_doc_data:
                self.tokenised_data.append(next_doc_data)
            
    def remove_empty_elements(self):
        self.tokenised_data = [d for d in self.tokenised_data if len(d) > 0]
            
    def get_next_batch(self):
        self.remove_empty_elements()
        self.fill_buffer()
        batch = [data.pop(0) for data in self.tokenised_data]
        return {'tokens': torch.stack([b['tokens'] for b in batch]),
                'targets': torch.stack([b['targets'] for b in batch]),}
