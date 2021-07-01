from torch.utils.data import Dataset, DataLoader
from pyramid_nested_ner.vectorizers.text.word import WordVectorizer
from pyramid_nested_ner.vectorizers.text.char import CharVectorizer
from pyramid_nested_ner.vectorizers.labels import PyramidLabelEncoder
from pyramid_nested_ner.data.sequence_bucketing import SequenceBucketing
from pyramid_nested_ner.utils.text import default_tokenizer

import torch


class PyramidNerDataset(Dataset):
    """
    Dataset class. Use its `get_dataloader` method to recover a DataLoader
    that can properly recover batches of samples from this Dataset, since
    this Dataset performs dynamic padding of tensors and collation is not
    straight-forward.
    """

    class _Collator(Dataset):

        def __init__(self, dataset, device='cpu'):
            self._device = device
            self._wrapped_dataset = dataset
            self._indices = torch.arange(len(dataset))

        def collate_fn(self, batch):
            batch = torch.stack(batch)
            actual_batch = dict()
            for name, tensors in self._wrapped_dataset[batch].items():
                if name == 'y':
                    actual_batch[name] = [tensor.to(self._device) for tensor in tensors]
                elif isinstance(batch[name], torch.Tensor):
                    actual_batch[name] = tensors.to(self._device)
                elif isinstance(batch[name], dict):
                    actual_batch[name] = self.collate_fn(
                                             batch[name])
            return actual_batch

        def __len__(self):
            return len(self._wrapped_dataset)

        def __getitem__(self, i):
            return self._indices[i]

    def __init__(
        self,
        data_reader,
        token_lexicon=None,
        custom_tokenizer=None,
        char_vectorizer=False,
        pyramid_max_depth=None
    ):
        """

        :param data_reader: generator of DataPoint objects representing the samples in the dataset.
        :param token_lexicon: iterable of strings containing the lexicon. If it's None, this will
                              be automatically generated from the data.
        :param custom_tokenizer: callable that performs tokenization given a single text input. If
                                 left to None, uses utils.text.default_tokenizer.
        :param char_vectorizer: adds char encodings.
        :param pyramid_max_depth: None for infinite depth.
        """

        self.data = [data_point for data_point in data_reader]
        self.tokenizer = custom_tokenizer or default_tokenizer

        if not token_lexicon:
            token_lexicon = {token for x in self.data for token in self.tokenizer(x.text)}

        self.word_vectorizer = WordVectorizer()
        self.word_vectorizer.set_tokenizer(self.tokenizer)
        self.word_vectorizer.fit(token_lexicon)
        if char_vectorizer:
            self.char_vectorizer = CharVectorizer()
            self.char_vectorizer.set_tokenizer(self.tokenizer)
            self.char_vectorizer.fit()
        else:
            self.char_vectorizer = None

        self.pyramid_max_depth = pyramid_max_depth
        self.label_encoder = PyramidLabelEncoder()
        self.label_encoder.set_tokenizer(self.tokenizer)
        self.label_encoder.fit([e.name for x in self.data for e in x.entities])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if isinstance(i, int):
            ids = torch.tensor([i])
            sample = [self.data[i]]
        else:
            indices = torch.arange(len(self.data)).long()
            sample = [self.data[index] for index in indices[i]]
            ids = torch.tensor([index for index in indices[i]])

        data = self._transform_x(sample)
        max_depth = self.pyramid_max_depth
        data['y'], data['y_remedy'] = self.label_encoder.transform(
                                       sample, max_depth=max_depth)
        data['id'] = ids.long()

        return data

    def _transform_x(self, sample):
        x = dict()
        for vect, role in [(self.word_vectorizer, 'word'), (self.char_vectorizer, 'char')]:
            if vect is not None:
                vectors, mask = vect.pad_sequences(vect.transform(sample))
                x[f'{role}_vectors'], x[f'{role}_mask'] = vectors, mask

        return x

    def get_dataloader(self, batch_size=32, shuffle=True, device='cpu', bucketing=False):

        def _collate_fn(batch, device=device):
            batch = batch[0]
            for name in batch.keys():
                if name == 'y':
                    batch[name] = [tensor.to(device) for tensor in batch[name]]
                elif isinstance(batch[name], torch.Tensor):
                    batch[name] = batch[name].to(device)
                elif isinstance(batch[name], dict):
                    batch[name] = _collate_fn([batch[name]], device)
            return batch

        if bucketing:  # use sequence bucketing
            sequence_lengths = torch.tensor([len(self.tokenizer(sample.text)) for sample in self.data])
            dataloader = SequenceBucketing.as_dataloader(self, sequence_lengths, batch_size, shuffle)
        else:
            collator = self._Collator(self, device)
            dataloader = DataLoader(collator, batch_size=batch_size, shuffle=shuffle)
            _collate_fn = collator.collate_fn
        dataloader.collate_fn = _collate_fn

        return dataloader
