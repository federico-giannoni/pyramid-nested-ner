import torch
from tqdm.autonotebook import tqdm


def load_bio_wv(wv_path):
    from genia.utils import wvlib
    wv = wvlib.load(wv_path)
    pbar = tqdm(total=None, desc='reading vectors')
    for word, vect in zip(wv.words(), wv.vectors()):
        yield word, torch.tensor(vect)
        pbar.update(1)
    pbar.close()


def get_bio_word_vectors(path):
    lexicon, weights = list(), list()
    for word, vect in load_bio_wv(path):
        lexicon.append(word)
        weights.append(vect)

    weights.insert(0, torch.zeros(200))  # pad token
    weights.append(torch.randn(200))  # unk token
    weights = torch.stack(weights)

    return lexicon, weights