import torch.nn as nn
import torch

from tqdm.autonotebook import tqdm


class PretrainedWordEmbeddings(nn.Module):

    def __init__(self, weights, padding_idx=0, freeze=True, **kwargs):
        super(PretrainedWordEmbeddings, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(weights,  padding_idx=padding_idx,  freeze=freeze,  **kwargs)

    @property
    def embedding_dim(self):
        return self.embeddings.embedding_dim

    @classmethod
    def from_csv(cls, csv, embedding_dim='infer', sep=" ", padding_idx=0, freeze=True, **kwargs):

        assert embedding_dim == 'infer' or isinstance(embedding_dim, int), f"Invalid embedding_dim {embedding_dim}."

        weights = list()
        lexicon = list()
        with open(csv, 'r') as fi:
            for line in tqdm(fi.readlines(), descr='reading...'):
                units = line.split(sep)
                token = units[0]
                vector = " ".join(units[1:])
                vector = [float(value) for value in vector.split()]
                if isinstance(embedding_dim, int):
                    if len(vector) == embedding_dim:
                        # we have a correct embedding;
                        weights.append(torch.tensor(vector))
                        lexicon.append(token)
                else:
                    weights.append(torch.tensor(vector))
                    lexicon.append(token)

        if embedding_dim == 'infer':
            dim_counts = dict()
            for weight in weights:
                dim_counts[weight.size(0)] = dim_counts.get(weight.size(0), 0) + 1
            embedding_dim = sorted(dim_counts.items(), key=lambda key_val: key_val[1])[-1][-1]
            weights = [weight for weight in weights if weight.size(0) == embedding_dim]

        weights.insert(0, torch.zeros(weights.size(-1)))
        weights.append(torch.randn(weights.size(-1)))
        weights = torch.stack(weights)

        return cls(weights, padding_idx=padding_idx, freeze=freeze, **kwargs), lexicon

    def forward(self, x):
        return self.embeddings(x)