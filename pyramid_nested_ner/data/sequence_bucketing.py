from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler

import torch


class SequenceBucketing(Dataset):
    """
    This class is designed to wrap a Dataset and organize its
    data in batches of similar length, so that padding can be
    reduced to a minimum for each batch, hence speeding up tr-
    aining. As of now, the class does not perform the padding
    itself (that is on the Dataset), but it can be extended
    to do so.
    """

    class _RandomSampler(RandomSampler):

        def __iter__(self):
            if not isinstance(self.data_source, SequenceBucketing):
                raise ValueError(
                    "SequenceBucketing._RandomSampler only "
                    "works with a SequenceBucketing dataset"
                )
            self.data_source.bucketize()  # shuffle data and rebuild buckets...
            yield from super(SequenceBucketing._RandomSampler, self).__iter__()

    def __init__(
        self,
        dataset: Dataset,
        sequence_lengths,
        bucket_size
    ):
        self._sequence_lengths = sequence_lengths
        self._wrapped_dataset = dataset
        self._bucket_size = bucket_size
        self._buckets = None
        self.bucketize()

    def bucketize(self, shuffle=False):
        """
        Creates buckets of indices, where each index represents a sample
        of the wrapped dataset. The buckets are created so that variance
        in total length in each bucket is minimized. Shuffling (at bucket
        level) can be performed if `shuffle` is True.
        :param shuffle: whether to shuffle the content of each bucket.
        :return:
        """
        sorted_lengths, sorted_indices = torch.sort(self._sequence_lengths)
        if shuffle:
            indices_by_length = dict()
            for length, index in zip(sorted_lengths.numpy(), sorted_indices.numpy()):
                if length not in indices_by_length:
                    indices_by_length[length] = list()
                indices_by_length[length].append(index)
            sorted_indices = torch.cat(
                [torch.tensor(indices)[torch.randperm(len(indices))] for indices in indices_by_length.values()])
        self._buckets = torch.split(sorted_indices, self._bucket_size)

    def __len__(self):
        return len(self._buckets)

    def __getitem__(self, i):
        if self._buckets is None:
            raise RuntimeError("Buckets haven't been built yet.")
        return self._wrapped_dataset[self._buckets[i]]

    @classmethod
    def as_dataloader(cls, dataset, sequence_lengths, batch_size=32, shuffle=True):
        """
        Automatically creates a DataLoader for the provided `dataset`, after applying
        sequence bucketing to it. It is recommended to keep `shuffle` to True to have
        a more stable gradient descent.
        :param dataset:
        :param sequence_lengths:
        :param batch_size:
        :param shuffle:
        :return:
        """
        sequence_bucketing = cls(dataset, sequence_lengths, batch_size)
        if not shuffle:
            sequence_bucketing.bucketize(shuffle=False)  # bucketize only once.
            dataloader = DataLoader(sequence_bucketing, batch_size=1, shuffle=False)
        else:
            dataloader = DataLoader(sequence_bucketing, batch_size=1,
                                    sampler=cls._RandomSampler(sequence_bucketing))
        return dataloader
