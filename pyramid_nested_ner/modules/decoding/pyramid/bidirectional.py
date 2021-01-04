import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pyramid_nested_ner.modules.decoding.pyramid import PyramidDecoder


class BidirectionalPyramidDecoder(PyramidDecoder):

    def __init__(self, input_size, hidden_size, max_depth=None, dropout=0.4):
        super(BidirectionalPyramidDecoder, self).__init__(input_size, hidden_size, max_depth, dropout)
        self.inverse_cnn = nn.Conv1d(
            in_channels=self.cnn.in_channels * 2,
            out_channels=self.cnn.in_channels,
            kernel_size=2,
            padding=1
        )

    def forward(self, x, mask):
        x, mask, original_indices = self._batchsort(x, mask)
        h = list()
        for h_l in self._pyramid_forward(x, mask):
            h.append(h_l)

        h_inverse = list()
        for h_l_inverse in self._inverse_pyramid_forward(h, mask):
            h_l_inverse = h_l_inverse[original_indices]
            h_inverse.append(h_l_inverse)
        h_inverse = h_inverse[::-1]  # reverse order!

        return self._separate_remedy_solution(h_inverse)

    def _inverse_pyramid_forward(self, h, mask):
        x_l = torch.zeros(*h[-1].size(), device=h[-1].device)
        for l_inv, h_l in enumerate(h[::-1]):
            l = len(h) - 1 - l_inv  # it's the corresponding index in the forward direction of the pyramid
            x_l, h_l_inverse = self._inverse_pyramid_layer_forward(h_l, x_l, mask[:, l:], is_last=(l == 0))
            yield h_l_inverse

    def _inverse_pyramid_layer_forward(self, h, x, mask, is_last=False):
        # remove empty sequences from computation
        non_empty_sequences = torch.sum(torch.sum(mask, dim=1) > 0).item()
        h, mask = h[:non_empty_sequences], mask[:non_empty_sequences]
        h = self.dropout(self.layer_norm(h))
        # forward to rnn and, eventually, to cnn
        batch_size = h.size(0)
        seq_length = h.size(1)
        sorted_lengths = torch.sum(mask, dim=1).cpu()
        self._init_hidden(batch_size, h.device)
        h = pack_padded_sequence(h, sorted_lengths, batch_first=self._is_bf)
        h, self.hidden = self.rnn(h, self.hidden)
        h, _ = pad_packed_sequence(
            h,
            total_length=seq_length,
            batch_first=self._is_bf)
        h = self.dropout(h)
        if h.size(0) < x.size(0):
            h = torch.cat((h, torch.zeros(x.size(0) - h.size(0), h.size(1), h.size(2), device=h.device)), dim=0)
        h = torch.cat((h, x), dim=-1)
        if is_last:
            x = None
        else:
            x = self.inverse_cnn(h.transpose(2, 1)).transpose(2, 1)

        return x, h