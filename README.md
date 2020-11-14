# Pyramid Nester NER

My implementation of the ACL 2020 paper ["Pyramid: A Layered Model for Nested Named Entity Recognition"](https://www.aclweb.org/anthology/2020.acl-main.525.pdf).

## Acknowledgements

To cite the paper:

```
@inproceedings{jue2020pyramid,
  title={Pyramid: A Layered Model for Nested Named Entity Recognition},
  author={Jue, WANG and Shou, Lidan and Chen, Ke and Chen, Gang},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={5918--5928},
  year={2020}
}
```

The implementation from the paper's authors is available at [this repository](https://github.com/LorrinWWW/Pyramid). I am not
the author of the paper, nor do I have any affiliation with the authors.

## Experiments

Since I could only recover a copy of the GENIA dataset online without having to pay, I was only able to replicate those experiments.
Look at the `genia` folder for a notebook that contains the code to download the data, word embeddings, BioBERT as well as training 
and evaluating the model.

## Additions

With respect to the original implementation I have added a couple of extra features, namely:

1. Support for [Flair](https://www.github.com/zalandoresearch/flair) static word embeddings out-of-the-box;
2. Support for end-to-end parsing (`model.parse_text`), as well as model saving and loading;
3. Extending the *remedy solution* of the paper in a multi-label setting, so that even with
   a small number of layers, the Pyramid model is still able to identify nested mentions longer than
   the pyramid's depth;
4. Implementing dynamic padding and sequence bucketing to speed up training.

## Train on your data

You can test the model on your own data easily through the `test/Pyramid Nested NER - Custom Training.ipynb` notebook. You can load the
Notebook on Google Colab and upload your data there (or on Drive) to train and evaluate easily. The notebook should also
give you an idea on how to export the model for inference. **It is recommended to use GPUs for training**.

For now, the only data format supported out-of-the-box is the older json format of rasa (shown below). You will
have to implement your own parsing logic for other formats. Look at the `pyramid_nested_ner/data/__init__.py`
file to see how data is modelled in terms of classes and attributes.

```json
"rasa_nlu_data": {
  "common_examples": [
    {
      "text": "Book a flight from Berlin to SF",
      "intent": "",
      "entities": [
        {
          "start": 19,
          "end": 25,
          "value": "Berlin",
          "entity": "city"
        },
        {
          "start": 29,
          "end": 31,
          "value": "San Francisco",
          "entity": "city"
        }
      ]
    }
  ]
}
```
