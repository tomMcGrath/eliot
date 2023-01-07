import datasets


def load_wikitext_train():
    """Loads the wikitext-103 train split"""
    dataset_group = 'wikitext'
    dataset_name = 'wikitext-103-raw-v1'
    dataset_split = 'train'
    dataset = datasets.load_dataset(
        dataset_group, name=dataset_name, split=dataset_split)
    return dataset


def get_text_data(ds_iterator):
    """WikiText-specific text iterator."""
    for record in ds_iterator:
        text = record['text'][0]
        if text:
            yield text