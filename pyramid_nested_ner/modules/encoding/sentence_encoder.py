import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
from pyramid_nested_ner.modules.rnn import FastRNN
from pyramid_nested_ner.modules.word_embeddings.transformer_embeddings import TransformerWordEmbeddings
from pyramid_nested_ner.modules.word_embeddings.flair_embeddings import FlairWordEmbeddings


class SentenceEncoder(nn.Module):

    def __init__(
        self,
        word_embeddings,
        char_embeddings=None,
        hidden_size=100,
        output_size=200,
        dropout=0.4,
        rnn_class=nn.LSTM,
        language_model=None  # type: TransformerWordEmbeddings
    ):
        input_size = word_embeddings.embedding_dim
        if char_embeddings is not None:   # using char embeddings
            input_size = input_size + char_embeddings.embedding_dim

        super(SentenceEncoder, self).__init__()
        self.word_embeddings = word_embeddings
        self.char_embeddings = char_embeddings
        self.rnn = FastRNN(
          rnn_class,
          input_size=input_size,
          hidden_size=hidden_size,
          batch_first=True,
          bidirectional=True
        )

        dense_input = self.rnn.hidden_size * 2
        if language_model is not None:
            dense_input += language_model.embedding_dim
        self.language_model = language_model
        self.dense = nn.Linear(dense_input, output_size)
        self.dropout = nn.Dropout(dropout or 0.0)

    def train(self, mode: bool = True):
        if isinstance(self.word_embeddings, FlairWordEmbeddings):
            self.word_embeddings.train(mode)
        if self.language_model is not None:
            self.language_model.train(mode)
        return super(SentenceEncoder, self).train(mode)

    def eval(self):
        if isinstance(self.word_embeddings, FlairWordEmbeddings):
            self.word_embeddings.eval()
        if self.language_model is not None:
            self.language_model.eval()
        return super(SentenceEncoder, self).eval()

    def _concat_char_and_word_embeddings(self, char_vectors, char_mask, word_vectors, word_mask):
        splits = torch.sum(word_mask, dim=1).tolist()
        char_vectors = self.char_embeddings(char_vectors,  char_mask)
        char_vectors = pad_sequence(torch.split(char_vectors, splits), batch_first=True)
        return torch.cat((char_vectors, word_vectors), dim=-1)

    def forward(self, word_vectors, word_mask, char_vectors=None, char_mask=None):
        x = self.word_embeddings(word_vectors)
        if all(element is not None for element in [self.char_embeddings, char_vectors, char_mask]):
            x = self._concat_char_and_word_embeddings(char_vectors, char_mask, x, word_mask)
        x = self.dropout(x)
        x, _ = self.rnn(x, word_mask)
        if self.language_model is not None:
            x_lm = self.language_model(word_vectors)
            x = torch.cat((x, x_lm), dim=-1)
        x = self.dense(x)

        return x, word_mask
