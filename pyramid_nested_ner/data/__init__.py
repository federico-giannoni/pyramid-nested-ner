import json


class Entity(object):

    def __init__(self, name, value, start, stop):
        self.name = name
        self.value = value
        self.start = start
        self.stop = stop

    def __str__(self):
        return json.dumps({
            'name': self.name,
            'value': self.value,
            'start': self.start,
            'stop': self.stop
        }, indent=2)

    def __eq__(self, other):
        if type(other) != type(self):
            return False

        return self.name == other.name and self.value == other.value and self.start == other.start and self.stop == other.stop


class DataPoint(object):

    def __init__(self, text, entities=None):
        self.text = text
        if not entities:
            entities = list()
        else:
            for entity in entities:
                assert isinstance(entity, Entity), f'{entity} is not an Entity.'
                if text[entity.start:entity.stop] != entity.value:
                    raise ValueError(
                        f'Misaligned entity: value is "{entity.value}", but in'
                        f' text the span is "{text[entity.start:entity.stop]}.'
                    )
        self.entities = entities

    def __str__(self):
        return json.dumps({
            'text': self.text,
            'entities': [
                {
                    'name': entity.name,
                    'value': entity.value,
                    'start': entity.start,
                    'stop': entity.stop
                }
                for entity in self.entities
            ]
        }, indent=2)

    def __eq__(self, other):
        if type(other) != type(self):
            return False

        if self.text != other.text:
            return False

        if len(self.entities) == len(other.entities):
            return all(entity in other.entities for entity in self.entities)
        return False
