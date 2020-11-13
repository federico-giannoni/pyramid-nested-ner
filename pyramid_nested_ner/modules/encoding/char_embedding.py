import torch
import torch.nn as nn

from pyramid_nested_ner.modules.rnn import FastRNN


class CharEmbedding(nn.Module):

    def __init__(self, lexicon, rnn=nn.GRU, embedding_dim=60, **kwargs):
        super(CharEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=len(lexicon),
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.rnn = FastRNN(
            rnn,
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            **kwargs
        )

    @property
    def embedding_dim(self):
        return self.rnn.hidden_size * (2 if self.rnn.bidirectional else 1)

    def forward(self, x, mask):
        """
        Forward a list of words through the char
        embedding RNN. Each char of each word is
        converted into a vector, that is passed
        through the RNN. The output is a tensor
        of word embeddings obtained from the cha-
        racter representations.
        """
        words = x.size(0)
        chars = x.size(1)
        lengths = torch.sum(mask, dim=1).long()

        # let BatchSize = BS, SequenceLength = SL, EmbeddingDim = ED.
        # remember that a sequence here is a word (sequence of chars)
        x = self.embedding(x)     # shape: [BS x SL x ED]
        x, _ = self.rnn(x, mask)  # shape: [BS x SL x rnn_h * rnn_directions]
        if self.rnn.bidirectional:
            x_forward, x_backward = self._separate_directions(x, lengths, words, chars)
            return torch.cat((x_forward, x_backward), dim=1)    # shape: [BS x 2*rnn_h]
        else:
            x_forward = x[torch.arange(words), lengths - 1, :]  # shape: [BS x 1*rnn_h]
            return x_forward

    def _separate_directions(self, x, lengths, words, chars):
        """
        Recovers the last element of the forward and backward
        directions and returns them, as separate [BS x rnn_h]
        tensors called `x_forward` and `x_backward`.
        :param x:
        :param lengths: non-padded lengths of the words
        :param words: number of words in x (the batch size)
        :param chars: number of chars in words (seq length)
        :return:
        """
        x = x.view(words, chars, 2, -1)
        x_forward = x[torch.arange(words), lengths - 1, 0, :]
        x_backward = x[:, 0, 1, :]
        return x_forward, x_backward
