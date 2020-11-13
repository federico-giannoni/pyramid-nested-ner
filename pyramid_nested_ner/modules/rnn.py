import torch.nn as nn
import torch

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FastRNN(nn.Module):
    """
    Generic class for RNN with support for dynamic batch lengths.
    It handles the packing and re-padding of sequences to not fe-
    ed padding tokens into the RNN.
    """

    def __init__(self, rnn_class, **kwargs):
        if rnn_class not in (nn.GRU, nn.LSTM):
            raise ValueError(
                f'Invalid RNN class type specified: must '
                f'be nn.GRU or nn.LSTM; got: {rnn_class}.'
            )
        super(FastRNN, self).__init__()
        self.bf = kwargs.get('batch_first', False)
        self.rnn = rnn_class(**kwargs)
        self.hidden = None

    @property
    def input_size(self):
        return self.rnn.input_size

    @property
    def bidirectional(self):
        return self.rnn.bidirectional

    @property
    def hidden_size(self):
        return self.rnn.hidden_size

    def _init_hidden(self, batch_size, device):
        d = (self.rnn.num_layers * 2 ** (int(self.rnn.bidirectional)), batch_size, self.rnn.hidden_size)
        if isinstance(self.rnn, nn.GRU):
            self.hidden = torch.zeros(d, device=device)
        if isinstance(self.rnn, nn.LSTM):
            self.hidden = torch.zeros(d, device=device), torch.zeros(d, device=device)

    def _batchsort(self, x, lengths):
        """
        Sorts the batch based on sequences' lengths.
        This step is required for pack_padded_seque-
        nce to work.
        :param x:
        :param lengths:
        :return:
        """
        sorted_lengths, sorted_ix = torch.sort(lengths, descending=True)
        _, unsorted_ix = torch.sort(sorted_ix)
        sorted_lengths = sorted_lengths.cpu()
        sorted_batch = x[sorted_ix]
        return sorted_batch, sorted_lengths, unsorted_ix

    def forward(self, x, mask):
        lengths = torch.sum(mask, dim=1)
        if not torch.max(lengths).item():
            raise RuntimeError("FastRNN got an empty sample.")
        batch_size = x.size(0)
        seq_length = x.size(1)
        self._init_hidden(batch_size, x.device)
        x, sorted_lengths, original_index = self._batchsort(x,  lengths)
        x = pack_padded_sequence(x, sorted_lengths, batch_first=self.bf)
        x, self.hidden = self.rnn(x, self.hidden)
        x, _ = pad_packed_sequence(x, total_length=seq_length, batch_first=self.bf)

        if isinstance(self.rnn, nn.GRU):
            self.hidden = self.hidden[:, original_index, :]
        else:
            h, c = self.hidden
            h, c = h[:, original_index, :], c[:, original_index, :]
            self.hidden = (h, c)

        return x[original_index], self.hidden
