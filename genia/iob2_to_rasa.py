from copy import deepcopy
from tqdm.autonotebook import tqdm


def iob_to_rasa(iob):
    for sentence_matrix in tqdm(iob.strip().split("\n\n")):
        tokens, spans = list(), list()  # parsing one sentence:
        for sentence_matrix_row in sentence_matrix.split("\n"):
            units = sentence_matrix_row.split("\t")
            spans.append(units[1:])
            tokens.append(units[0])

        spans = [list(tup) for tup in zip(*spans)]

        entities = list()
        for span in spans:
            offset = 0
            for i in range(0, len(tokens)):
                label = span[i]
                if label.startswith('B'):
                    j = i + 1
                    while j < len(tokens) and span[j].startswith('I'):
                        j += 1
                    entity = {
                        'entity': label[2:],
                        'tokens': list(range(i, j)),
                        'value': " ".join([tokens[k] for k in range(i, j)]),
                        'start': offset,
                    }
                    entity['end'] = entity['start'] + len(entity['value'])
                    entities.append(deepcopy(entity))
                offset += len(tokens[i]) + 1

        yield {
            'text': " ".join(tokens),
            'entities': deepcopy(entities),
            'intent': "",
        }


if __name__ == "__main__":
    import os
    import sys
    import json
    source, target = sys.argv[1:]

    for fil in os.listdir(source):
        filepath = os.path.join(source, fil)
        with open(filepath, 'r') as f:
            rasa_data = {
                'rasa_nlu_data': {
                    'intent_examples': [],
                    'entity_examples': [],
                    'common_examples': [
                        example for example in iob_to_rasa(f.read())
                    ]
                }
            }

        json.dump(rasa_data, open(os.path.join(target, fil.replace("iob2", "rasa")), 'w'), indent=2)
        print(f'converted {os.path.join(source, fil)} to {os.path.join(target, fil.replace("iob2", "rasa"))}')
