from pyramid_nested_ner.utils.text import default_tokenizer
from torch.nn.utils.rnn import pad_sequence

import torch


class PyramidLabelEncoder(object):
    """
    Label encoder class responsible for transforming entity annotations
    into torch.Tensors. The `transform` API returns two tensors: one fo-
    r the pyramid layer and a second one for the so-called 'remedy solu-
    tion', that uses IOB2 notation in a multi-label setting.
    """

    default_tokenizer = default_tokenizer

    def __init__(self):
        self.entities = list()

    def tokenize(self, text):
        return self.default_tokenizer(text)

    def set_tokenizer(self, tokenizer: callable):
        self.default_tokenizer = tokenizer

    @property
    def iob2_entities(self):
        return [f'{iob2}-{entity}' for entity in self.entities for iob2 in 'IB' if entity]

    def fit(self, entities):
        self.entities = [None, *{entity for entity in entities}]

    def transform(self, data, max_depth=None):
        """
        Transforms the (list of) DataPoint(s) into tensors. More precisely, this method
        returns two items: a list containing one tensor for each layer up to `max_depth`,
        a second tensor that encodes entities using the 'remedy solution'.  The tensors
        in the first list enumerate all possible spans in `data` and assign a single cl-
        ass to each span, so each tensor in the list has a shape B x S x 1, where B is
        the "batch size" (the number of samples in `data`) and S is the maximum sequen-
        ce length (in tokens). The remedy solution instead has shape B x S x 2 * C, wh-
        ere C are the unique entities that were passed to `fit` (entity lexicon). Tens-
        ors in the first list are to be used with a `CrossEntropyLoss`, while the reme-
        dy solution is in multi-label format and requires `BCEWithLogitsLoss`.
        :param data:
        :param max_depth: if None, max_depth = inf, therefore remedy solution will be None.
        :return:
        """
        y_layers = list()
        y_remedy = None
        if max_depth is not None:
            for layer in range(max_depth):
                y_layer = self._transform_layer(data, layer + 1)
                if y_layer.numel():
                    y_layers.append(y_layer)
                else:
                    break
            if len(y_layers) == max_depth:
                # use remedy solution to cover entities longer than `max_depth`
                y_remedy = self._remedy_encoding_transform(data, max_depth + 1)
                if not y_remedy.numel():
                    y_remedy = None
        else:
            while 1:
                y_layer = self._transform_layer(data, len(y_layers) + 1)
                if y_layer.numel():
                    y_layers.append(y_layer)
                else:
                    break
        return y_layers, y_remedy

    def _ngrams(self, text, order):
        start = 0
        tokens = self.tokenize(text)
        while start + order <= len(tokens):
            yield tokens[start:start + order]
            start = start + 1

    def _entity_ngram_bitmap(self, data_point, order):
        for i, ngram in enumerate(self._ngrams(data_point.text, order)):
            ngram_start = i
            ngram_stop = i + len(ngram)
            valid_candidate = False
            for entity in data_point.entities:
                entity_start = len(self.tokenize(data_point.text[:entity.start]))
                entity_stop = entity_start + len(self.tokenize(entity.value))
                if entity_start == ngram_start and entity_stop == ngram_stop:
                    yield self.entities.index(entity.name)
                    valid_candidate = True
                    break  # a span can only have one annotation
            if not valid_candidate:
                yield 0

    def _remedy_solution_bitmap(self, data_point, order):
        remedy_solution_bitmap = list()
        for i, ngram in enumerate(self._ngrams(data_point.text, order)):
            ngram_start = i
            ngram_stop = i + len(ngram)
            ngram_bitmap = torch.zeros(len(self.iob2_entities))
            for entity in data_point.entities:
                entity_start = len(self.tokenize(data_point.text[:entity.start]))
                entity_stop = entity_start + len(self.tokenize(entity.value))
                if ngram_start >= entity_start and ngram_stop <= entity_stop:
                    # current n-gram is inside an entity span:
                    if entity_start == ngram_start:
                        iob2_entity = f'B-{entity.name}'
                    elif ngram_stop <= entity_stop:
                        iob2_entity = f'I-{entity.name}'
                    else:
                        raise AssertionError(" ")
                    ngram_bitmap[self.iob2_entities.index(iob2_entity)] = 1
            remedy_solution_bitmap.append(ngram_bitmap.clone())
        if remedy_solution_bitmap:
            return torch.stack(remedy_solution_bitmap)
        return torch.tensor([])

    def _remedy_encoding_transform(self, data, order):
        y_layer = list()
        for x in (data if isinstance(data, list) else [data]):
            y_layer.append(self._remedy_solution_bitmap(x, order))
        try:
            return pad_sequence(y_layer, batch_first=True)
        except RuntimeError:
            # pad_sequence can crash if some sequences in `data` are shorter than the number of lay-
            # ers and therefore their encoding yields an empty tensor, while other sequences are tr-
            # ansformed into tensors of shape (n, |iob2_entities|).
            y_layer = [y if y.numel() else torch.zeros(1, len(self.iob2_entities)) for y in y_layer]
            return pad_sequence(y_layer, batch_first=True)

    def _transform_layer(self, data, layer):
        y_layer = list()
        for x in (data if isinstance(data, list) else [data]):
            y_layer.append(torch.tensor([bit for bit in self._entity_ngram_bitmap(x, layer)]))
        return pad_sequence(y_layer, batch_first=True).long()

    def inverse_transform(self, y):
        return [self._inverse_layer_transform(y_layer) for y_layer in y]

    def inverse_remedy_transform(self, y_remedy):

        def _recover_span(tensor_slice, entity_name):
            for j, vect in enumerate(tensor_slice[1:]):
                if not vect[self.iob2_entities.index(f'I-{entity_name}')]:
                    return tensor_slice[:j + 1]
            return tensor_slice

        longest_span, sequences_tags = 0, list()

        for sequence in y_remedy:
            sequence_tags = dict()
            for offset, logits in enumerate(sequence):
                for entity in self.entities[1:]:
                    if logits[self.iob2_entities.index(f'B-{entity}')]:
                        span = _recover_span(sequence[offset:], entity)
                        if len(span) not in sequence_tags:
                            sequence_tags[len(span)] = ['O' for _ in range(len(sequence) - (len(span) - 1))]
                        if 'O' == sequence_tags[len(span)][offset]:
                            sequence_tags[len(span)][offset] = f'B-{entity}'
                            longest_span = max(len(span), longest_span)
                        else:
                            sequence_tags[len(span)][offset] = None
                            # print(
                            #   f"Tokens {span} have two different annotations: "
                            #   f"{sequence_tags[len(span)][2:]}, and {entity}. "
                            #   f"Due to this conflict, both annotations will be"
                            #   f" discarded."
                            # )
            sequences_tags.append(sequence_tags)

        decoded_labels = list()
        for i in range(1, longest_span + 1):
            decoded_labels_for_order = list()
            for sequence, sequence_tags in zip(y_remedy, sequences_tags):
                sequence_length = max(0, len(sequence) - (i - 1))
                if i in sequence_tags:
                    span = [iob2_tag or 'O' for iob2_tag in sequence_tags[i]]
                else:
                    span = ['O' for _ in range(sequence_length)]
                decoded_labels_for_order.append(span)
            decoded_labels.append(decoded_labels_for_order)

        return decoded_labels

    def _inverse_layer_transform(self, y_layer):
        sequences_tags = list()
        for sequence in y_layer:
            tags = [f'B-{self.entities[index]}' if index else 'O' for index in sequence]
            sequences_tags.append(tags)
        return sequences_tags
