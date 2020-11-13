from pyramid_nested_ner.data import DataPoint, Entity

import json


def rasa_data_reader(path):
    for example in json.load(open(path, 'r'))['rasa_nlu_data']['common_examples']:
        entities = [
          Entity(
            entity['entity'],
            entity['value'],
            entity['start'],
            entity['end']
          ) for entity in example['entities']
        ]

        yield DataPoint(example['text'], entities)
