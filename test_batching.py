import tiktoken
import torch
import batching
import wikitext_utils


class TestDataSource:

    def test_data_flow(self):
        """End-to-end test of the data loader/batcher."""
        dataset = wikitext_utils.load_wikitext_train()
        text_iterator = wikitext_utils.make_shuffled_text_iterator(dataset)
        encoding = tiktoken.get_encoding('gpt2')

        seq_len = 10
        batch_size = 5
        process_fn = batching.make_processor(encoding.encode, seq_len)
        datasources = [batching.DataSource(text_iterator, process_fn) 
                       for _ in range(batch_size)]
        batched_data_source = batching.BatchedDataSource(datasources)
        batched_data = batched_data_source.get_next()
        assert batched_data['tokens'].shape == torch.Size([batch_size, seq_len])
        assert batched_data['tokens'].shape == batched_data['targets'].shape