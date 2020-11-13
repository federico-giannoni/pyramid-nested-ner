import torch

from flair.data import Sentence
from tqdm.autonotebook import tqdm
from flair.embeddings import WordEmbeddings, StackedEmbeddings


class FlairWordEmbeddings(object):

    def __init__(self, embeddings, lexicon, padding_idx=0, dropout=None, device='cpu'):

        self._names = []
        if isinstance(embeddings,  list) and len(embeddings) > 1:
            flair_embeddings = list()
            for embedding in embeddings:
                self._names.append(embedding)
                flair_embeddings.append(WordEmbeddings(embedding))
            self.embeddings = StackedEmbeddings(flair_embeddings)
        elif isinstance(embeddings, list):
            self._names.append(embeddings[0])
            self.embeddings = WordEmbeddings(embeddings[0])
        else:
            self._names.append(embeddings)
            self.embeddings = WordEmbeddings(embeddings)

        self.lexicon = lexicon
        self.dropout = dropout
        self.device = device
        self.embeddings = self.embeddings.to(device)
        self.pad_index = padding_idx
        self._train = True

    def to(self, device, **kwargs):
        self.embeddings = self.embeddings.to(device, **kwargs)
        return self

    def train(self, mode=True):
        self._train = mode

    def eval(self):
        self._train = False

    @property
    def word_embeddings_names(self):
        return self._names

    @property
    def vocab_idx(self):
        return {i: token for i, token in enumerate(self.lexicon)}

    @property
    def embedding_dim(self):
        return self.embeddings.embedding_length

    def _embed(self, x):
        vocab_idx = self.vocab_idx
        embeddings = []
        for sequence in x:
            padding_length = sequence.size(0)
            flair_sentence = Sentence()
            for index in sequence:
                index = index.item()
                if index == self.pad_index:
                    break  # skip padding
                padding_length = padding_length - 1
                token = vocab_idx.get(index, '[UNK]')
                flair_sentence.add_token(token)
            self.embeddings.embed(flair_sentence)
            sentence_embedding = torch.stack([token.embedding for token in flair_sentence.tokens])
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

        return torch.stack(embeddings)

    def precompute(self, sentences, out_file, sep=" ", use_tokenizer=None):
        """
        Pre-computes the word embeddings for your dataset and persists the
        lexicon and weights to `out_file` as a csv. This can be useful be-
        cause flair embeddings are slower than using a simple nn.Embedding
        module from torch. If you want faster performance, you can use th-
        is method and then use PretrainedEmbeddings.from_csv to load back
        the lexicon and weights from the csv file (look at the file
        word_embeddings.pretrained_embeddings.py).
        :param sentences:
        :param out_file:
        :param sep:
        :param use_tokenizer:
        :return:
        """
        seen_tokens = set()
        with open(out_file, 'w') as fi:
            for sentence in tqdm(sentences, descr=' embedding...'):
                flair_sentence = Sentence(sentence, use_tokenizer=use_tokenizer)
                self.embeddings.embed(flair_sentence)
                for token in flair_sentence:
                    token, embedding = token.text, token.embedding
                    if token not in seen_tokens:
                        embedding = embedding.cpu().numpy().tolist()
                        fi.write(f"{token}{sep}{' '.join(embedding)}\n")
                        seen_tokens.add(token)

        print(f"Done. Precomputed {len(seen_tokens)} embeddings at {out_file}.")

    def __call__(self, x):
        x = self._embed(x)
        x = torch.dropout(x, p=self.dropout or 0.0, train=self._train)
        return x
