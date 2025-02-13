from custom.datasets import MatchChatDataset, add_team_player_tokens, add_msep, point_seperate_messages
from datasets import DatasetDict, Dataset 
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForMaskedLM, TrainingArguments, Trainer, TrainerCallback, TrainerState
from datetime import datetime
import torch
import json
import bisect
import random
import pandas as pd


pd_df = # load data here

matches = list(set(pd_df['match']))
random.shuffle(matches)
test_matches = set(matches[:1000])
train_matches = set(matches[1000:])

test_df = pd_df[pd_df['match'].isin(test_matches)]
train_df = pd_df[pd_df['match'].isin(train_matches)]

# set player and team tokens depending on max players and teams in the dataset
team_player_tokens = ['[Player'+str(i)+']' for i in range(20)] + ['[Team'+str(i)+']' for i in range(7)]




preproc_num = 16
batch_size = 16
lr=5e-5
epochs = 10
model_name = 'distilbert/distilroberta-base'
tokenizer_name = model_name
message_concatenator = # set dialog representation: add_team_player_tokens, add_msep, ...
training_name = #set output name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
##################### add special tokens ##################### 
special_tokens = team_player_tokens
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
model.resize_token_embeddings(len(tokenizer))
##################### add special tokens ##################### 
token_summary = {'additional_special_tokens': special_tokens, 'tokenizer': tokenizer_name}

output_name = model_name.split('/')[-1]+ '_' + training_name
model_output_path = "models/mlm/{}-{}".format(output_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
print(json.dumps({'path': model_output_path, 'message_concatenator': message_concatenator.__name__,'initial_learning_rate': lr, 'batch_size': batch_size, 'epochs': epochs} | token_summary))
dataset = DatasetDict({
    'train': Dataset.from_dict({'text': MatchChatDataset(chat_df=train_df, message_concatenator=message_concatenator, team_tokens=True)}), 
    'test': Dataset.from_dict({'text': MatchChatDataset(chat_df=test_df, message_concatenator=message_concatenator, team_tokens=True)})
    })

def preprocess_function(examples):
    return tokenizer(examples["text"])

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=preproc_num,
    remove_columns=dataset["train"].column_names,
)

block_size = tokenizer.model_max_length # 512 for distilroberta-base
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size: # shorten total lenth to the nearest multiple of block size
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result




lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=preproc_num)
batches_per_epoch = len(lm_dataset['train'])//batch_size
print(json.dumps({'samples per epoch': len(lm_dataset['train']), 'batches per epoch': batches_per_epoch}))
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


training_args = TrainingArguments(
    output_dir=model_output_path,
    learning_rate=lr,
    num_train_epochs=epochs,
    weight_decay=0.01,
    save_steps=batches_per_epoch,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # logging_dir='logs',
    logging_steps=batches_per_epoch
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    # eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
    # callbacks=[my_callback()]
)

trainer.train()