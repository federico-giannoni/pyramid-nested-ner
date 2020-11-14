import torch
import flair
import os


class TransformerWordEmbeddings(object):
    """
    Fake-module to build word embeddings from transformers' token embeddings.
    Any model from huggingface transformers library can be loaded (using its
    name or path to model weights), if it's supported by Flair. Keep in mind
    that this class is extremely slow and should only be used for research
    experiments. It uses a cache to speed up all training epochs following
    the first one, but inference time on unseen samples remains prohibitive.
    """

    def __init__(
            self,
            transformer,
            lexicon,
            padding_idx=0,
            layers="-1,-2",
            pooling_operation='mean',
            dropout=2e-1,
            device='cpu',
            casing=True,
    ):
        try:
            self.embeddings = flair.embeddings.TransformerWordEmbeddings(
                transformer,
                layers=layers,
                fine_tune=False,
                pooling_operation=pooling_operation,
                use_scalar_mix=True
            )
        except OSError:  # try from_tf=True
            self.embeddings = flair.embeddings.TransformerWordEmbeddings(
                transformer,
                layers=layers,
                fine_tune=False,
                pooling_operation=pooling_operation,
                use_scalar_mix=True,
                from_tf=True
            )

        if os.path.isdir(transformer):
            self._transformer = os.path.abspath(transformer)
        else:
            self._transformer = transformer

        self.pad_index = padding_idx
        self.lexicon = lexicon
        self.dropout = dropout
        self._train = True
        self._cache = dict()
        self.device = device
        self.casing = casing
        self.embeddings = self.embeddings.to(device)

    def to(self, device, **kwargs):
        self.embeddings = self.embeddings.to(device, **kwargs)
        return self

    def train(self, mode=True):
        self._train = mode

    def eval(self):
        self._train = False

    @property
    def transformer(self):
        return self._transformer

    @property
    def vocab_idx(self):
        return {i: token.lower() if not self.casing else token for i, token in enumerate(self.lexicon)}

    @property
    def embedding_dim(self):
        return self.embeddings.embedding_length

    def __tensor_to_cache_key(self, tensor):
        tensor = tensor[:torch.sum(tensor != self.pad_index)]
        cache_key = " ".join([str(value.item()) for value in tensor.cpu()])
        return cache_key

    def _get_from_cache(self, key):
        key = self.__tensor_to_cache_key(key)
        return self._cache.get(key)

    def _add_to_cache(self, key, value):
        key = self.__tensor_to_cache_key(key)
        self._cache[key] = value.cpu()

    def _embed(self, x):
        vocab_idx = self.vocab_idx
        embeddings = []
        for sequence in x:
            padding_length = sequence.size(0)
            if self._get_from_cache(sequence) is None:
                flair_sentence = flair.data.Sentence()
                for index in sequence:
                    index = index.item()
                    if index == self.pad_index:
                        break  # skip padding
                    padding_length = padding_length - 1
                    token = vocab_idx.get(index, '[UNK]')
                    flair_sentence.add_token(token)
                self.embeddings.embed(flair_sentence)
                sentence_embedding = torch.stack([token.embedding.cpu() for token in flair_sentence.tokens])
                self._add_to_cache(sequence, sentence_embedding)
            else:
                sentence_embedding = self._get_from_cache(sequence)
                padding_length = padding_length - sentence_embedding.size(0)
            if padding_length:
                sentence_embedding = torch.cat((
                    sentence_embedding,
                    torch.zeros(
                        padding_length,
                        sentence_embedding.size(-1),
                        device=sentence_embedding.device
                    )
                ))
            embeddings.append(sentence_embedding)

        return torch.stack(embeddings).to(self.device)

    def __call__(self, x):
        x = self._embed(x)
        x = torch.dropout(x, p=self.dropout or 0.0, train=self._train)
        return x
