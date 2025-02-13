# imports
from datasets import (
    LabeledChatReplayDataset,
    current_player_only_point_sep,
    point_seperate_messages,
    add_msep,
    current_player_only_msep,
    int_label,
    team_player_tokens,
    msep_token,
    add_team_player_tokens
)

import jsonlines as jsonl
import numpy as np
from datetime import datetime
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from sklearn.utils.class_weight import compute_class_weight
import random
import math



logfilepath = 'metrics.jsonl'
save_model_filepath = '../models/within_match_experiments/'
# static hyperparameters (not included in hyperparameter search)
# hyperparams
# lr = 5e-6
epochs = 20
batch_size = 64
# message_concatenator = current_player_only_point_sep
# additional_special_tokens = []
# base_model_name = 'distilbert/distilroberta-base'
runs = 10
weight_decay = .01
momentum = .0


# define wighted loss model, compute metrics
class WeightedLossModel(torch.nn.Module):
    def __init__(self, model, class_weights) -> None:
        super().__init__()
        self.model = model
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to('cuda'))

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        if labels is not None:
            loss = self.loss_fct(outputs.logits, labels)
            return loss, outputs.logits
        else:
            return outputs.logits
    
    def save_pretrained(self, path):
        self.model.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path, class_weights):
        return cls(AutoModelForSequenceClassification.from_pretrained(path), class_weights)

def get_metric_dict(y_true, y_pred_prob, thresh=.5):
    y_pred = np.array(y_pred_prob) > thresh
    return {'acc': float(accuracy_score(y_true, y_pred)), 'bal_acc': float(balanced_accuracy_score(y_true, y_pred)), 'prec': float(precision_score(y_true, y_pred)), 'rec': float(recall_score(y_true, y_pred)), 'bin_f1': float(f1_score(y_true, y_pred)), 'auc': float(roc_auc_score(y_true, y_pred_prob))}

# define proper batching, torch format, to cuda
def stratify_batch_iter(to_batch, batch_size):
    class1 = [t for t in to_batch if t[1]]
    class0 = [t for t in to_batch if not t[1]]
    # sample stratified batches
    batches = []
    while class1+class0:
        ratio = len(class1)/len(class1+class0)
        batch = class1[:math.floor(ratio*batch_size)] + class0[:math.ceil((1-ratio)*batch_size)]
        class1, class0 = class1[math.floor(ratio*batch_size):], class0[math.ceil((1-ratio)*batch_size):]
        random.shuffle(batch)
        batches.append(batch)
    random.shuffle(batches)

    for batch in batches:
        inputs, labels = list(zip(*batch))
        assert len(labels) > 1 # not defined behaviour for single, this is for batch
        # go from list of dict to dict of list
        inputs = {k: [d[k] for d in inputs] for k in inputs[0]}
        input_ids = torch.vstack(inputs['input_ids'])
        attention_mask = torch.vstack(inputs['attention_mask'])
        yield input_ids.to('cuda'), attention_mask.to('cuda'), torch.tensor(labels).to('cuda')


# hyperparam search loop
hyperparams = [
    # ['distilbert/distilroberta-base', point_seperate_messages, [], 7e-6],
    # ['distilbert/distilroberta-base', current_player_only_point_sep, [], 4e-6],
    # ['../models/mlm/distilroberta-base_blob_input-2024-05-14_21-02-48/checkpoint-267930', point_seperate_messages, [], 1e-5],
    # ['../models/mlm/distilroberta-base_blob_input-2024-05-14_21-02-48/checkpoint-267930', current_player_only_point_sep, [], 4e-6],
    # ['../models/mlm/distilroberta-base_special_msep-2024-06-08_07-06-29/checkpoint-217390', add_msep, [msep_token], 7.5e-6],
    # ['../models/mlm/distilroberta-base_special_msep-2024-06-08_07-06-29/checkpoint-217390', current_player_only_msep, [msep_token], 4e-6],
]
for params in hyperparams:
    experiment_start = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')
    base_model_name, message_concatenator, additional_special_tokens, lr = params

    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilroberta-base')
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    if additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
    tokenize = lambda x: tokenizer(x, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

    train_dset = LabeledChatReplayDataset(dataset_name='train_labels', message_concatenator=message_concatenator, cache=True, tokenizer=tokenize, format_label=int_label).cache
    test_dset = LabeledChatReplayDataset(dataset_name='test_labels', message_concatenator=message_concatenator, cache=True, tokenizer=tokenize, format_label=int_label).cache
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=np.array(list(zip(*train_dset))[1])).tolist()
    print('loaded and tokenized datasets, train class weights:', class_weights)


    for run in range(runs):
        model = WeightedLossModel(AutoModelForSequenceClassification.from_pretrained(base_model_name), class_weights=class_weights).to('cuda')

        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=22)

# make metadata dict
        metadata = {
            'run': run,
            'run_start': datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M'),
            'experiment_start': experiment_start,
            'total_runs': runs,
            'base_model': base_model_name,
            'model': str(model),
            'lr': lr,
            'epochs': epochs,
            'batch_size': batch_size,
            'message_concatenator': message_concatenator.__name__,
            'additional_special_tokens': additional_special_tokens,
            'optimizer': str(optimizer),
            'lr_scheduler': str(lr_scheduler),
        }
        print(metadata)

# train test loop
        for epoch in range(epochs):
            random.shuffle(train_dset)
            random.shuffle(test_dset)
            train_dataloader = list(stratify_batch_iter(train_dset, batch_size))
            test_dataloader = list(stratify_batch_iter(test_dset, batch_size))
            model.train()
            total_loss = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
                loss, outputs = model(*batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                all_pred_probs = []
                all_labels = []
                total_test_loss = 0

                for test_batch in test_dataloader:
                    input_ids, attention_mask, labels = test_batch
                    all_labels += labels.cpu()

                    test_loss, test_outputs = model(input_ids, attention_mask, labels)
                    total_test_loss += test_loss.item()
                    all_pred_probs.append(softmax(test_outputs.cpu(), axis=1)[:, 1])

            epoch_trl = total_loss/len(train_dataloader)
            epoch_tel = total_test_loss/len(test_dataloader)
            metrics = get_metric_dict(all_labels, np.concatenate(all_pred_probs, axis=0)) | {'train_loss': epoch_trl, 'test_loss': epoch_tel, 'epoch': epoch}

            with jsonl.open(logfilepath, 'a') as f:
                f.write({'metadata': metadata, 'metrics': metrics})
            print(f"Epoch {epoch}/{epochs}, Training Loss: {epoch_trl}, Test Loss: {epoch_tel}")

            model.save_pretrained(save_model_filepath+metadata['experiment_start']+'-run_'+str(run)+'/epoch_'+str(epoch))