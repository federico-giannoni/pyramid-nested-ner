from pyramid_nested_ner.utils.text import default_tokenizer


class TextVectorizer(object):
    """
    Base class for text vectorization. `transform` turns a (list of)
    sentence(s) into a (list of) tensor(s) and `pad_sequences` appl-
    ies padding to obtain a single tensor to be used as a batch.
    """

    default_tokenizer = default_tokenizer

    PAD = '<PAD>'
    UNK = '<UNK>'

    def fit(self, *args, **kwargs):
        pass

    def transform(self, data, **kwargs):
        pass

    def tokenize(self, text):
        return self.default_tokenizer(text)

    def set_tokenizer(self, tokenizer: callable):
        self.default_tokenizer = tokenizer

    def fit_transform(self, data, **kwargs):
        self.fit(**kwargs)
        return self.transform(data, **kwargs)

    def pad_sequences(self, data, **kwargs):
        pass

    def render(self, data, **kwargs):
        pass
