import torch.nn as nn
import torch
import yaml
import os

from pyramid_nested_ner.modules.encoding import CharEmbedding, SentenceEncoder
from pyramid_nested_ner.modules.word_embeddings.flair_embeddings import FlairWordEmbeddings
from pyramid_nested_ner.modules.word_embeddings.transformer_embeddings import TransformerWordEmbeddings
from pyramid_nested_ner.modules.decoding.pyramid.bidirectional import PyramidDecoder, BidirectionalPyramidDecoder
from pyramid_nested_ner.modules.decoding.linear import LinearDecoder
from pyramid_nested_ner.vectorizers.text.word import WordVectorizer
from pyramid_nested_ner.vectorizers.text.char import CharVectorizer
from pyramid_nested_ner.vectorizers.labels import PyramidLabelEncoder
from pyramid_nested_ner.data import DataPoint, Entity


class PyramidNer(object):

    class _Model(nn.Module):

        def __init__(self, encoder, pyramid, classifier):
            super(PyramidNer._Model, self).__init__()
            self.encoder = encoder
            self.pyramid = pyramid
            self.classifier = classifier

        def forward(self, *args, **kwargs):
            x, mask = self.encoder(*args, **kwargs)
            h, h_remedy = self.pyramid(x, mask)
            return self.classifier(h, h_remedy)

    def __init__(
        self,
        word_lexicon,
        word_embeddings,
        entities_lexicon,
        language_model=None,
        language_model_casing=True,
        char_embeddings_dim=60,
        encoder_hidden_size=100,
        encoder_output_size=200,
        decoder_hidden_size=100,
        pyramid_max_depth=None,
        batch_first=True,
        inverse_pyramid=False,
        custom_tokenizer=None,
        decoder_dropout=0.4,
        encoder_dropout=0.4,
        device='cpu'
    ):
        if isinstance(word_embeddings, str):
            word_embeddings = [word_embeddings]
        if isinstance(word_embeddings, list) and isinstance(word_embeddings[0], str):
            word_embeddings = FlairWordEmbeddings(
                word_embeddings,
                lexicon=word_lexicon,
                padding_idx=0
            )

        self._model_args = {
            'word_embeddings': word_embeddings.to(device),
            'language_model': language_model,
            'char_embeddings_dim': char_embeddings_dim,
            'encoder_hidden_size': encoder_hidden_size,
            'encoder_output_size': encoder_output_size,
            'decoder_hidden_size': decoder_hidden_size,
            'pyramid_max_depth': pyramid_max_depth,
            'batch_first': batch_first,
            'inverse_pyramid': inverse_pyramid,
            'decoder_dropout': decoder_dropout,
            'encoder_dropout': encoder_dropout,
            'device': device
        }

        self.tokenizer = custom_tokenizer or (lambda t: t.split())
        self.label_encoder = PyramidLabelEncoder()
        self.word_vectorizer = WordVectorizer()
        self.char_vectorizer = CharVectorizer()
        self.label_encoder.fit(entities_lexicon)
        self.word_vectorizer.fit(word_lexicon)
        self.char_vectorizer.fit()
        for component in [self.word_vectorizer, self.char_vectorizer, self.label_encoder]:
            component.set_tokenizer(self.tokenizer)
        if language_model is not None:
            self._model_args['language_model'] = TransformerWordEmbeddings(
                language_model,
                word_lexicon,
                padding_idx=0,
                device=device,
                casing=language_model_casing
            )

        self.nnet = self._init_nnet()

    @property
    def device(self):
        return self._model_args['device']

    def reset_weights(self):
        self.nnet = self._init_nnet()
        print('New model created!')

    def logits_to_classes(self, logits):
        return [torch.argmax(nn.functional.softmax(logit, dim=-1), dim=-1) for logit in logits]

    def remedy_to_classes(self, logits):
        if logits is None:
            return
        return torch.round(torch.sigmoid(logits))

    def classes_to_iob2(self, classes, remedy=None):
        labels = self.label_encoder.inverse_transform(classes)
        if remedy is not None:
            labels += self.label_encoder.inverse_remedy_transform(remedy)
        return labels

    def parse(self, x):
        if isinstance(x, list):
            return [self._parse_text(text) for text in x]
        return self._parse_text(x)

    def _parse_text(self, text):
        assert isinstance(text, str), f'Can not parse {text} (not a string).'
        x = " ".join(self.tokenizer(text))  # tokenization
        device = self.device
        x_word, word_mask = self.word_vectorizer.pad_sequences(self.word_vectorizer.transform([DataPoint(x)]))
        x_char, char_mask = self.char_vectorizer.pad_sequences(self.char_vectorizer.transform([DataPoint(x)]))

        self.nnet.eval()
        with torch.no_grad():
            layers, remedy = self.nnet(x_word.to(device), word_mask.to(device), x_char.to(device), char_mask.to(device))
        self.nnet.train(mode=True)

        layers_classes = self.logits_to_classes(layers)
        remedy_classes = self.remedy_to_classes(remedy)
        labels = self.classes_to_iob2(layers_classes, remedy=remedy_classes)

        entities = list()
        tokens = x.split()
        for l, layer in enumerate(labels):
            assert len(layer) == 1
            sequence = layer[0]
            for token, tag in enumerate(sequence):
                if tag == 'O':
                    continue
                entity = tag[2:]  # discard the IOB2 notation
                value = " ".join(tokens[token:token + l + 1])
                stop = len(" ".join(tokens[:token + l + 1]))
                start = stop - len(value)
                entities.append(Entity(entity, value, start, stop))

        return DataPoint(x, entities)

    def _build_char_embeddings(self, char_embeddings_dim):
        if not char_embeddings_dim:
            return None
        if char_embeddings_dim % 2:
            raise ValueError(
                f'Dimension of character embeddings must be '
                f'an even number (got {char_embeddings_dim})'
            )
        return CharEmbedding(
            self.char_vectorizer.X,
            rnn=nn.LSTM,
            bidirectional=True,
            embedding_dim=int(char_embeddings_dim / 2),
            batch_first=self._model_args['batch_first'],
        )

    def _init_nnet(self):
        sentence_encoder = SentenceEncoder(
            self._model_args['word_embeddings'],
            char_embeddings=self._build_char_embeddings(self._model_args['char_embeddings_dim']),
            hidden_size=self._model_args['encoder_hidden_size'],
            output_size=self._model_args['encoder_output_size'],
            rnn_class=nn.LSTM,
            language_model=self._model_args['language_model'],
            batch_first=self._model_args['batch_first'],
            dropout=self._model_args['encoder_dropout'],
        )

        if self._model_args['inverse_pyramid']:
            pyramid_cls = BidirectionalPyramidDecoder
        else:
            pyramid_cls = PyramidDecoder
        pyramid_decoder = pyramid_cls(
            input_size=self._model_args['encoder_output_size'],
            hidden_size=self._model_args['decoder_hidden_size'],
            batch_first=self._model_args['batch_first'],
            dropout=self._model_args['decoder_dropout'],
            max_depth=self._model_args['pyramid_max_depth']
        )
        decoder_output_size = self._model_args['decoder_hidden_size'] * 2 * (
                1 + int(self._model_args['inverse_pyramid']))

        classifier = LinearDecoder(decoder_output_size, classes=len(self.label_encoder.entities))
        return self._Model(sentence_encoder, pyramid_decoder, classifier).to(self.device)

    def save(self, path, name='pyramid_ner'):
        """
        Saves the model weights and metadata to the specified path,
        so it can be loaded later using the `load` class method.
        :param path:
        :param name:
        :return:
        """
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory.")

        folder = os.path.join(path,  name)
        os.makedirs(folder, exist_ok=True)

        model_metadata = self._model_args
        if isinstance(self._model_args['word_embeddings'], FlairWordEmbeddings):
            model_metadata['word_embeddings'] = self._model_args['word_embeddings'].word_embeddings_names
        else:
            embedding_layer = model_metadata['word_embeddings']
            model_metadata['word_embeddings'] = {
                'num_embedding': embedding_layer.num_embeddings,
                'embedding_dim': embedding_layer.embedding_dim,
                'padding_idx': embedding_layer.padding_idx,
                'freeze': not embedding_layer.requires_grad
            }
        if isinstance(self._model_args['language_model'], TransformerWordEmbeddings):
            model_metadata['language_model'] = self._model_args['language_model'].transformer
        # persist metadata
        yaml.safe_dump(model_metadata, open(os.path.join(folder, 'metadata.yml'), 'w'))

        # persist token lexicon
        with open(os.path.join(folder, 'lexicon.txt'), 'w') as lex:
            for token in self.word_vectorizer.lexicon:
                lex.write(f"{token}\n")
        # persist label lexicon
        with open(os.path.join(folder, 'entities.txt'), 'w') as en:
            for entity in self.label_encoder.entities:
                if entity is not None:
                    en.write(f"{entity}\n")
        state_dict = self.nnet.state_dict()
        # persist state_dict (model weight)
        torch.save(state_dict, os.path.join(folder, 'weights.bin'))

    @classmethod
    def load(cls, path, custom_tokenizer=None, force_device=None, force_language_model=None, force_embeddings=None):
        if not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory.")
        try:
            model_metadata = yaml.safe_load(path)
        except Exception as e:
            raise ValueError(f"Could not load 'metadata.yml' file at {path}: {e}")

        model_metadata['device'] = force_device or model_metadata['device']
        model_metadata['language_model'] = force_language_model or model_metadata['language_model']
        model_metadata['word_embeddings'] = force_embeddings or model_metadata['word_embeddings']

        if isinstance(model_metadata['word_embeddings'], dict):
            # rebuild the word embeddings matrix (the weights will be loaded from the state_dict)
            freeze = model_metadata['word_embeddings'].pop('freeze')
            place_holder = nn.Embedding(**model_metadata['word_embeddings'])
            place_holder.weight.requires_grad = not freeze
            model_metadata['word_embeddings'] = place_holder

        with open(os.path.join(path, 'lexicon.txt'), 'r') as lex:
            lexicon = [token for token in lex.read().split('\n') if token]
        with open(os.path.join(path, 'entities.txt'), 'r') as en:
            entities = [entity for entity in en.read().split('\n') if entity]

        kwargs = model_metadata
        kwargs['word_lexicon'] = lexicon
        kwargs['entities_lexicon'] = entities
        kwargs['custom_tokenizer'] = custom_tokenizer
        obj = cls(**kwargs)

        state_dict = torch.load(os.path.join(path, 'weights.bin'))
        obj.nnet.load_state_dict(state_dict)
        return obj
