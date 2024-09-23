from toxicity_detection.custom.datasets import LabeledMultiSourceDataset, UserMatchHistoryDataset, LabeledChatReplayDataset, current_player_only_special_tokens, point_seperate_messages, current_player_only_point_sep, team_player_tokens, add_msep, msep_token, current_player_only_msep, GameEventDataset, TorchDatasetGetItemWrapper, ChatReplayDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from scipy.special import softmax
import jsonlines as jsonl
from datetime import datetime
import sys
import pandas as pd
import matplotlib.pyplot as plt
from transformers import RobertaModel, RobertaTokenizer, AutoModelForSequenceClassification, AutoTokenizer, pipeline

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import one_hot


embed_model_tokenizer = AutoTokenizer.from_pretrained('distilbert/distilroberta-base')
embed_model_tokenizer.padding_side = 'left'
embed_model_tokenizer.truncation_side = 'left'

class dset_wrapper(Dataset):
  def __init__(self, dset):
    super().__init__()
    self.dset = dset

  def __getitem__(self, idx):
    return {k: v[idx] for k, v in self.dset.items()}

  def __len__(self):
    return len(self.dset[list(self.dset.keys())[0]])

def get_cls_embedding(embed_model, text):
    # Tokenize the input text
    inputs = embed_model_tokenizer(text, return_tensors='pt', truncation=True, padding=True).to('cuda')
    # Get the outputs from the model
    with torch.no_grad():
        outputs = embed_model(**inputs, output_hidden_states=True)
    # The hidden states are in the form of a tuple where the last element is the hidden state of the [CLS] token
    hidden_states = outputs.hidden_states  # A tuple: (layers, batch_size, sequence_length, hidden_size)
    # The first token of the sequence is the [CLS] token
    cls_embeddings = hidden_states[-1][:, 0, :]  # Shape: (batch_size, hidden_size)
    return cls_embeddings

def get_embeds(embed_model, text, batch_size=64):
  # batch and vstack embeds
  return torch.vstack([get_cls_embedding(embed_model, text[i:i+batch_size]) for i in range(0, len(text), batch_size)])


# get embeddings of current (labeled) messages, and other messages of these players from other matches
train_dsets = []
test_dsets = []
for run in range(10): # the embedding model was trained 10 times to get average metrics
    modelpath = 'runs/2024-09-15 17:16-run_'+str(run)+'/epoch_4' # epoch 4 is the best epoch for this experiment
    embed_model = AutoModelForSequenceClassification.from_pretrained(modelpath).to('cuda')

    train_chat_df = # private data
    test_chat_df = # private data
    train_text, train_labels = list(zip(*LabeledChatReplayDataset(chat_df=train_chat_df, dataset_name=None, message_concatenator=current_player_only_point_sep)))
    test_text, test_labels = list(zip(*LabeledChatReplayDataset(chat_df=test_chat_df, dataset_name=None, message_concatenator=current_player_only_point_sep)))
    # maybe pytorch format labels
    train_embeds = get_embeds(embed_model, train_text)
    test_embeds = get_embeds(embed_model, test_text)

    unlabeled_chat_df = pd.read_csv('unlabeled_match_chats.csv')
    unlabeled_chat_df['message'] = unlabeled_chat_df['message'].apply(lambda x: str(x))
    unlabeled_text = list(ChatReplayDataset(chat_df=unlabeled_chat_df, dataset_name=None, message_concatenator=current_player_only_point_sep))
    unlabeled_senders = unlabeled_chat_df['user_id'].tolist()
    # make map of user to mean embed
    user_embed_map = {user: get_embeds(embed_model, [t for t, u in zip(unlabeled_text, unlabeled_senders) if u==user]).mean(axis=0) for user in set(unlabeled_senders)}
    # vstack[map[u] for u in chat_df['user_id']
    train_user_embeds = torch.vstack([user_embed_map[u] for u in train_chat_df['user_id']])
    test_user_embeds = torch.vstack([user_embed_map[u] for u in test_chat_df['user_id']])
    # torch hstack sources
    train_dataset = dset_wrapper({'input': torch.hstack([train_embeds, train_user_embeds]), 'label': one_hot(torch.tensor(train_labels, dtype=torch.int64), num_classes=2).float().to('cuda')})
    test_dataset = dset_wrapper({'input': torch.hstack([test_embeds, test_user_embeds]), 'label': one_hot(torch.tensor(test_labels, dtype=torch.int64), num_classes=2).float().to('cuda')})

    train_dsets.append(train_dataset)
    test_dsets.append(test_dataset)


# Compute class weights for cost-sensitive loss
# print('mean label:', sum(train_labels)/len(train_labels))
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
# class_weights = [0.53909304, 6.895]
# print('class weights:', class_weights)
class_weights = torch.tensor(class_weights, dtype=torch.float)


# Custom classification model
class MultiSourceModel(nn.Module):
    def __init__(self, class_weights):
        super(MultiSourceModel, self).__init__()
        self.class_weights = class_weights.to('cuda')
        self.classifier = nn.Sequential( # multiple network architectures were tested, but on average none performed much better than baseline
        nn.Linear(2*768, 2*768),
        nn.Dropout(0.1),
        nn.Linear(2*768, 2),
        )

    def forward(self, input_embeds, labels=None):
        logits = self.classifier(input_embeds)
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
# define metrics to log
def get_metric_dict(y_true, y_pred_prob, thresh=.5):
    y_pred = np.array(y_pred_prob) > thresh
    return {'threshold': float(thresh), 'acc': float(accuracy_score(y_true, y_pred)), 'bal_acc': balanced_accuracy_score(y_true, y_pred), 'prec': float(precision_score(y_true, y_pred)), 'rec': float(recall_score(y_true, y_pred)), 'bin_f1': float(f1_score(y_true, y_pred)), 'auc': float(roc_auc_score(y_true, y_pred_prob))}



weight_decay = .01
momentum = .0
for lr in [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4]*10: # hyperparameter search, + repeat 10 times to get average metrics
    num_epochs = 20
    for run in range(10): # iterate over all 10 runs of the embedding model
        train_dataset = train_dsets[run]
        test_dataset = test_dsets[run]
        batch_size = len(train_text)
        sources = [modelpath+' embed', 'mean other match embed']
        run_start = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        model = MultiSourceModel(class_weights).to('cuda')
        metadata = {
            'batch_size': batch_size,
            'lr': lr,
            'num_epochs': num_epochs,
            'class_weights': [float(w) for w in class_weights],
            'sources': sources,
            'model_summary': str(model),
            'run_start': run_start,
            'optimizer': str(optimizer)
        }


        # Initialize optimizer, lr scheduler
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs+2, eta_min=0)  # eta_min is the minimum learning rate

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
            
                loss, outputs = model(input_embeds=batch['input'], labels=batch['label'])

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Step the scheduler at the end of each epoch
            scheduler.step()
            # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(train_dataloader)}")

            # Calculate test accuracy
            model.eval()
            correct_predictions = 0
            total_predictions = 0
            with torch.no_grad():
                all_pred_probs = []
                all_labels = []
                for test_batch in test_dataloader:
                    labels = [int(l[1]) for l in test_batch["label"]]
                    all_labels += labels
                    
                    batch_outputs = model(input_embeds=test_batch['input'])
                    batch_pred_prob = softmax(batch_outputs.cpu(), axis=1)[:, 1]
                    all_pred_probs.append(batch_pred_prob)

                metric_dict = get_metric_dict(all_labels, np.concatenate(all_pred_probs, axis=0))
                # print(metric_dict['bal_acc'])
                with jsonl.open('metrics.jsonl', 'a') as f:
                    f.write({'metadata': metadata, 'metrics': metric_dict | {'epoch': epoch, 'train_loss': total_loss/len(train_dataloader)}})