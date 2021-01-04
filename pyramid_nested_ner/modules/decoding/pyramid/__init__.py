import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PyramidDecoder(nn.Module):
    """
    The pyramid module. In the paper, https://www.aclweb.org/anthology/2020.acl-main.525.pdf,
    the writers only specify that the RNN weights are shared across all decoding layer, howe-
    ver, in this implementation, also the CNN weights used by each pyramid layer are the same.
    Having separate Conv1d modules for each layer would only train each conv module to identi-
    fy entity of a fixed length. If the length of the entities in the dataset is not well dis-
    tributed, this would yield suboptimal training for some of the convolutional modules.
    """

    def __init__(self, input_size, hidden_size, max_depth=None, dropout=0.4):
        super(PyramidDecoder, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        channels = hidden_size * 2
        self.cnn = nn.Conv1d(in_channels=channels, out_channels=channels,  kernel_size=2)
        self.dropout = nn.Dropout(dropout or 0.)
        self.layer_norm = nn.LayerNorm(channels)
        self._max_depth = max_depth
        self._is_bf = True
        self.hidden = None

    @property
    def max_depth(self):
        return self._max_depth

    def _batchsort(self, x, mask):
        lengths = torch.sum(mask, dim=1)
        sorted_lengths, sorted_indices = torch.sort(lengths, descending=True)
        _, original_indices = torch.sort(sorted_indices, descending=False)
        return x[sorted_indices], mask[sorted_indices], original_indices

    def _init_hidden(self, batch_size, device):
        self.hidden = (
            torch.zeros(2, batch_size, self.rnn.hidden_size, device=device),
            torch.zeros(2, batch_size, self.rnn.hidden_size, device=device),
        )

    def _separate_remedy_solution(self, h):
        if self._max_depth is not None and len(h) > self._max_depth:
            return h[:-1] or None, h[-1]
        else:
            return h, None

    def forward(self, x, mask):
        x, mask, original_indices = self._batchsort(x, mask)
        h = list()
        for h_l in self._pyramid_forward(x, mask):
            h_l = h_l[original_indices]  # restore original order
            h.append(h_l)

        return self._separate_remedy_solution(h)

    def _pyramid_forward(self, x, mask):
        """
        Before moving data through the pyramid layers, we sort the
        batch by length so we can pack the padding elements and al-
        so remove altogether from the batch the sequences that do
        not have any element left. Indeed, if a batch contains one
        sample with length 5 and one sample with length 10, the nu-
        mber of convlution operations appliable to both is not the
        same (we can get up to 5 convs on the sample with length 5,
        and up to 10 on the sample with length 10). Therefore, we
        progressively hide shorter sequences as they shrink in si-
        ze one convolution after another.

        The batch is restored at the end by adding zero tensors in
        place of removed samples. This should speed up training as
        we only compute operations for the samples for which they
        are needed.

        :param x:
        :param mask:
        :return:
        """
        batch_size = x.size(0)
        seq_length = x.size(1)
        if self._max_depth is not None:
            max_depth = self._max_depth + 1  # add remedy solution
        else:
            max_depth = seq_length
        for l in range(max_depth):
            x, h_l = self._pyramid_layer_forward(x, mask[:, l:])
            if h_l.size(0) < batch_size:  # re-add empty sequences
                h_l = torch.cat(
                    (h_l, torch.zeros(batch_size - h_l.size(0), h_l.size(1), h_l.size(2), device=h_l.device)), dim=0)
            yield h_l
            if x is None:
                break

    def _pyramid_layer_forward(self, x, mask):
        # remove empty sequences from computation
        non_empty_sequences = torch.sum(torch.sum(mask, dim=1) > 0).item()
        x, mask = x[:non_empty_sequences], mask[:non_empty_sequences]
        x = self.dropout(self.layer_norm(x))
        # forward to rnn and, eventually, to cnn
        batch_size = x.size(0)
        sorted_lengths = torch.sum(mask, dim=1).cpu()
        self._init_hidden(batch_size, x.device)
        h = pack_padded_sequence(x, sorted_lengths, batch_first=self._is_bf)
        h, self.hidden = self.rnn(h, self.hidden)
        h, _ = pad_packed_sequence(h, total_length=x.size(1),
                                   batch_first=self._is_bf )
        h = self.dropout(h)
        if torch.max(sorted_lengths).item() > 1:
            x = self.cnn(h.transpose(2, 1)).transpose(2, 1)
        else:
            x = None
        return x, h
