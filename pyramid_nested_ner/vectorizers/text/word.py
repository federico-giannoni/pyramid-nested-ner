from pyramid_nested_ner.vectorizers.text import TextVectorizer
from torch.nn.utils.rnn import pad_sequence
import torch


class WordVectorizer(TextVectorizer):

    def __init__(self):
        self.X = dict()

    @property
    def unk(self):
        return self.X.get(self.UNK)

    @property
    def lexicon(self):
        return [word for word in self.X.keys() if word not in [self.PAD, self.UNK]]

    @property
    def index_lookup(self):
        return {v: k for k, v in self.X.items()}

    def fit(self, lexicon, **kwargs):
        self.X = {self.PAD: 0}
        for word in lexicon:
            self.X[word] = len(self.X)
        self.X[self.UNK] = len(self.X)

    def transform(self, data, **kwargs):
        return [self._transform_sentence(sentence) for sentence in (data if isinstance(data, list) else [data])]

    def _transform_sentence(self, data_point):
        return torch.tensor([self.X.get(token, self.unk) for token in self.tokenize(data_point.text)]).long()

    def pad_sequences(self, data, **kwargs):
        mask = pad_sequence([torch.ones(sequence.size(0), dtype=torch.int8) for sequence in data], batch_first=True)
        return pad_sequence(data, batch_first=True), mask

    def render(self, tensor, **kwargs):
        for vector in tensor:
            print(" ".join([self.index_lookup[index.item()] for index in vector]))
