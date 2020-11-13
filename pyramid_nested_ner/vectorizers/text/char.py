from pyramid_nested_ner.vectorizers.text import TextVectorizer
from torch.nn.utils.rnn import pad_sequence
import string
import torch


class CharVectorizer(TextVectorizer):

    PAD = '□'

    def __init__(self, fit=False):
        self.X = dict()
        if fit:
            self.fit()

    @property
    def index_lookup(self):
        return {v: k for k, v in self.X.items()}

    def fit(self, **kwargs):
        self.X = {self.PAD: 0}
        for char in string.printable:
            self.X[char] = len(self.X)
        self.X[self.UNK] = len(self.X)

    def transform(self, data, **kwargs):
        return [
            self._transform_words(data_point) for data_point in (data if isinstance(data,  list) else [data])
        ]

    def _transform_words(self, data_point, **kwargs):
        return [
            torch.tensor([self.X[char] for char in token]).long() for token in self.tokenize(data_point.text)
        ]

    def pad_sequences(self, data, return_splits=False, **kwargs):
        char_vectors = [char_vector for sentence_matrix in data for char_vector in sentence_matrix]
        mask = pad_sequence([torch.ones(vect.size(0)) for vect in char_vectors],  batch_first=True)
        if not return_splits:
            return pad_sequence(char_vectors, batch_first=True), mask
        else:
            return pad_sequence(char_vectors, batch_first=True), mask, [len(sentence) for sentence in data]

    def render(self, tensor, splits=None):
        if isinstance(tensor, torch.Tensor):
            tensor = torch.split(tensor, splits or len(tensor))
        for matrix in tensor:
            tokens = ["".join([self.index_lookup[index.item()] for index in vector]) for vector in matrix]
            print(" ".join(tokens))
