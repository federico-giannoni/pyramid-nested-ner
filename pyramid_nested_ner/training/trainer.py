from pyramid_nested_ner.model import PyramidNer

from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from seqeval.metrics.sequence_labeling import classification_report
from copy import deepcopy

import torch.nn as nn
import pandas as pd
import numpy as np
import torch


class PyramidNerTrainer(object):

    class _TrainingReport(object):

        def __init__(self):
            self._total = 0
            self.epochs = {
                'train_loss': [],
                'valid_loss': [],
            }

        def add_epoch(self, train_loss, valid_loss=None, **kwargs):
            self.epochs['train_loss'].append(train_loss)
            self.epochs['valid_loss'].append(valid_loss)
            for key, arg in kwargs.items():
                self.epochs[key] = self.epochs.get(key, list())
                self.epochs[key].append(arg)
            self._total += 1

        @property
        def report(self):
            return pd.DataFrame(index=np.arange(self._total) + 1, data=self.epochs)

        def plot_loss_report(self):
            self.plot_custom_report('train_loss', 'valid_loss')

        def plot_custom_report(self, *columns):
            xticks = [i for i in self.report.index if not i % 10 or i == 1]
            self.report[[*columns]].plot(xticks=xticks, xlabel='epoch')

    def __init__(self, pyramid_ner: PyramidNer):
        self._optimizer = None
        self._scheduler = None
        self._model = pyramid_ner
        self.device = self._model.device

    @property
    def nnet(self):
        return self._model.nnet

    @property
    def optim(self):
        return self._optimizer

    @property
    def ner_model(self):
        return self._model

    def train(
        self,
        train_set: DataLoader,
        optimizer,
        epochs=10,
        patience=np.inf,
        dev_data: DataLoader = None,
        scheduler=None,
        grad_clip=None,
        restore_weights_on='loss'  # 'loss' to restore weights with best dev loss, 'f1' for best dev f1
    ):
        if patience is None:
            patience = 0
        train_report = self._TrainingReport()
        self._optimizer = optimizer
        self._scheduler = scheduler
        overall_patience = patience

        if restore_weights_on not in ['loss', 'f1', None]:
            raise ValueError(
                f"Param 'restore_weights_on' can only be 'loss' or 'f1', depending on which"
                f" is the preferred metric for weight restoration. Got {restore_weights_on}"
            )
        best_dev_f1, best_dev_loss = 0.0, np.inf
        best_weights = {'loss': None, 'f1': None}

        for epoch in range(epochs):
            print('==============================')
            print(f'Training epoch {epoch + 1}...')
            print('==============================')
            train_loss = self._training_epoch(train_set, grad_clip)
            if self._scheduler:
                self._scheduler.step()
            if dev_data is not None:
                report = self.test_model(dev_data, out_dict=True)
                micro_f1 = report['micro avg']['f1-score'] * 100
                dev_loss = report['loss']
                train_report.add_epoch(
                    train_loss, dev_loss, micro_f1=micro_f1
                )
                if dev_loss < best_dev_loss or micro_f1 > best_dev_f1:  # good epoch!!
                    overall_patience = patience
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        best_weights['loss'] = deepcopy(self.nnet.state_dict())
                    if micro_f1 > best_dev_f1:
                        best_dev_f1 = micro_f1
                        best_weights[ 'f1' ] = deepcopy(self.nnet.state_dict())
                elif patience < np.inf:
                    print(f'Bad epoch... (patience left: {overall_patience}/{patience})')
                    if not overall_patience:
                        print('Stopping early (restoring best weights)...')
                        self.nnet.load_state_dict(best_weights)
                        break
                    overall_patience -= 1
            else:
                train_report.add_epoch(train_loss, None)

        if restore_weights_on and best_weights[restore_weights_on] is not None:
            print('Training is done (restoring model\'s best weights)')
            self.nnet.load_state_dict(best_weights[restore_weights_on])

        return self._model, train_report

    def _training_epoch(self, train_set, grad_clip):
        train_loss = list()
        pbar = tqdm(total=len(train_set))
        for batch in train_set:
            # forward
            y, remedy_y, ids = batch.pop('y'), batch.pop('y_remedy'), batch.pop('id')
            self._optimizer.zero_grad()
            logits, remedy = self.nnet(**batch)
            # loss computation
            loss = self.compute_loss(logits, y, batch['word_mask'], remedy, remedy_y)
            train_loss.append(loss.item())
            # backward
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), grad_clip)
            pbar.set_description(f'train loss: {round(np.mean(train_loss), 2)}')
            self._optimizer.step()
            pbar.update(1)
        pbar.close()
        return np.mean(train_loss)

    def test_model(self, test_data, out_dict=False):
        loss = list()
        pred, true = list(), list()
        pbar = tqdm(total=len(test_data))
        for batch in test_data:
            # inference
            self.nnet.eval()
            y, remedy_y, ids = batch.pop('y'), batch.pop('y_remedy'), batch.pop('id')
            with torch.no_grad():
                logits, remedy = self.nnet(**batch)
            self.nnet.train(mode=True)
            # loss computation
            batch_loss = self.compute_loss(logits, y, batch['word_mask'], remedy, remedy_y).item()
            loss.append(batch_loss)
            layers_y_hat = self._model.logits_to_classes(logits)
            remedy_y_hat = self._model.remedy_to_classes(remedy)
            pbar.set_description(f'valid loss: {round(np.mean(loss), 3)}')
            y_pred, y_true = self._classes_to_iob2(layers_y_hat, y, True, remedy_y_hat,  remedy_y)
            pred.extend(y_pred)
            true.extend(y_true)
            pbar.update(1)

        report = self.classification_report(pred, true, out_dict)
        if out_dict:
            report['loss'] = np.mean(loss)
            f1_score = round(report["micro avg"]["f1-score"] * 100, 2)
            pbar.set_description(
                f'valid loss: {round(np.mean(loss), 2)}; micro f1: {f1_score}%'
            )
        pbar.close()

        return report

    @staticmethod
    def compute_loss(logits, y, mask, remedy_logits=None, remedy_y=None) -> torch.Tensor:
        assert len(logits) == len(y), 'Predictions and labels are misaligned.'
        if remedy_y is None or remedy_logits is None:
            assert remedy_y is None and remedy_logits is None, 'Predictions and labels are misaligned'
        cross_entropy = nn.CrossEntropyLoss(reduction='none')
        loss = 0.0
        for i, (logits_layer, y_layer) in enumerate(zip(logits, y)):
            layer_loss = cross_entropy(logits_layer.permute(0, -1, 1), y_layer)
            loss += torch.sum(layer_loss * mask[:, i:])

        if remedy_y is not None and remedy_logits is not None:
            ml_loss = nn.BCEWithLogitsLoss(reduction='none')(remedy_logits, remedy_y)
            ml_mask = mask[:, len(logits):].unsqueeze(-1).expand_as(ml_loss)
            loss += torch.sum(ml_loss * ml_mask)

        # note that we return the sum of the loss of each token, rather than averaging it;
        # average leads to a loss that is too small and generates small gradients that pr-
        # event the model from learning anything due to its depth.

        return loss

    def _classes_to_iob2(self, pred, true, flatten=False, remedy_pred=None, remedy_true=None):
        y_pred = self._model.classes_to_iob2(pred, remedy=remedy_pred)
        y_true = self._model.classes_to_iob2(true, remedy=remedy_true)
        if len(y_pred) > len(y_true):
            y_true.extend([[['O' for _ in y] for y in extra_layer] for extra_layer in y_pred[len(y_true):]])
        if len(y_true) > len(y_pred):
            y_pred.extend([[['O' for _ in y] for y in extra_layer] for extra_layer in y_true[len(y_pred):]])
        if flatten:
            y_pred = [seq for layer in y_pred for seq in layer]
            y_true = [seq for layer in y_true for seq in layer]

        return y_pred, y_true

    @staticmethod
    def classification_report(y_pred, y_true, out_dict=False):
        from seqeval.scheme import IOB2
        report = classification_report(
            y_true,
            y_pred,
            digits=4,
            output_dict=out_dict,
            mode='strict',
            zero_division=0,
            scheme=IOB2
        )
        return report
