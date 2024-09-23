from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from collections import Counter
import random
import warnings

msep_token = '<msep>'
def add_msep(messages: list, players=None) -> str:
    return msep_token.join(messages)
def current_player_only_msep(messages: list, players: list) -> str:
    return add_msep(current_player_only(messages, players))

def get_special_token(player: int, relative_to: int) -> str:
    assert player <= 9 and player >= 0
    assert relative_to <= 9 and relative_to >= 0
    team_token = '[Team' + str((int(player<5)-int(relative_to<5))%2) + ']'
    player_token = '[Player' + str((player - relative_to)%5) + ']'

    return team_token + player_token

team_player_tokens = [
    '[Team0]',
    '[Team1]',
    '[Player0]',
    '[Player1]',
    '[Player2]',
    '[Player3]',
    '[Player4]',
    ]

def current_player_only_special_tokens(messages: list, players: list) -> str:
    m = current_player_only(messages, players)
    p = [0]*len(m)
    return add_special_tokens(m, p)

def add_special_tokens(messages: list, players: list, relative_to=-1) -> str:
    if relative_to != None: 
        relative_to = players[relative_to]
    # else message not sent by a player (e.g. game anouncement)
    return '\n'.join([get_special_token(players[i], relative_to) + ' ' + messages[i] for i in range(len(messages))])

def current_player_only(messages: list, players: list) -> list:
    assert len(messages) == len(players)
    return [messages[i] for i in range(len(messages)) if players[i]==players[-1] or not players[i]]

def point_seperate_messages(messages: list, players=None) -> str:
    return '\n'.join([m+'. ' for m in messages])

def current_player_only_point_sep(m: list, p: list):
    return point_seperate_messages(current_player_only(m, p))

def current_message_only(messages: list, players=None) -> str:
    return messages[-1]


bool_label = lambda label: (label == 'x') or (label == 'c')
int_label = lambda label: int(bool_label(label))

class TorchDatasetGetItemWrapper(Dataset):
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._get_slice(idx)
        subset = self._get_subset(idx)
        if subset != None:
            return subset
        return self._getitem(idx)
        
    def _get_slice(self, s):
        return [self._getitem(ii) for ii in range(*s.indices(len(self)))]

    def _get_subset(self, indices):
        try:
            indices = list(indices)
        except TypeError:
            indices = None
        if indices != None:
            return [self._getitem(i) for i in indices]
        else:
            return None


class ChatDataFrameSelector(TorchDatasetGetItemWrapper):
    def __init__(self, chat_df: pd.DataFrame) -> None:
        super().__init__()
        # doctor index to skip messages not from players
        chat_df = chat_df.fillna(np.nan).replace([np.nan], [None])
        chat_df['new_i'] = chat_df['player'].apply(lambda x: int(x!=None)).cumsum() - 1
        chat_df['new_i'][chat_df['new_i'].duplicated()] = -1
        self.chat_df = chat_df.set_index('new_i')
        self.match_dict = {i: sub_df for i, sub_df in self.chat_df.groupby('match')}#[['message', 'player']]} why limit this?
        self.chat_match_index_map = {j:i for i, sub_df in self.match_dict.items() for j in sub_df.index if j!=None}

    def _getitem(self, index) -> pd.DataFrame:
        # skip game events (player==None)
        try:
            return self.match_dict[self.chat_match_index_map[index]].loc[:index]
        except KeyError as e:
            raise IndexError() from e # should raise index error and not key error in order to be used like a list

    def __len__(self):
        # only include indeces of messages sent by players (not game events)
        return sum((self.chat_df['player'].apply(lambda x: int(x!=None))).tolist())


class ChatReplayDataset(ChatDataFrameSelector):
    def __init__(self, dataset_name='romovpa', message_concatenator=add_special_tokens, chat_df=None, tokenizer=None):
        if chat_df is None:
            if dataset_name == 'romovpa':
                chat_df = pd.read_csv('../data/dota2/romovpa/dota2_chat_messages.csv', index_col=False).dropna(subset=['text', 'match', 'slot']).drop_duplicates(['match', 'text', 'time', 'slot']).reset_index()
                chat_df.rename(columns={'text': 'message', 'slot': 'player'}, inplace=True)
            elif dataset_name == 'devinanzelmo':
                chat_df = pd.read_csv('../data/dota2/devinanzelmo/chat.csv', index_col=False).dropna(subset=['key', 'match_id', 'slot']).drop_duplicates(['match_id', 'key', 'time', 'slot']).reset_index()
                chat_df.drop(chat_df[chat_df['slot'] == -9].index, inplace=True) # remove some wierd data
                chat_df.rename(columns={'key': 'message', 'match_id': 'match', 'slot': 'player'}, inplace=True)
                chat_df.reset_index(inplace=True) 
            else:
                raise NotImplementedError('unknown dataset: "'+str(dataset_name)+'"')

        super().__init__(chat_df)
        self.concat_messages = message_concatenator
        self.tokenizer = tokenizer
        
    def _getitem(self, index) -> str:
        selected_df = super()._getitem(index)
        text = self.concat_messages(selected_df['message'].to_list(), selected_df['player'].to_list())
        if self.tokenizer:
            return self.tokenizer(text)
        else:
            return text
    

class LabeledChatReplayDataset(ChatReplayDataset):
    def __init__(self, 
                 dataset_name='train_labels', 
                 message_concatenator=add_special_tokens, 
                 chat_df=None, 
                 format_label=bool_label, 
                 label_shift_type=None, 
                 label_shift_n=None, 
                 tokenizer=None, 
                 cache=False):
        
        if dataset_name:
            chat_df = pd.read_csv('/media/storage/adrien/toxicity-detection/'+dataset_name+'.csv', index_col=0).fillna(np.nan).replace([np.nan], [None]).reset_index()
        if label_shift_type:
            assert type(label_shift_n) == int and label_shift_n > 0
            if label_shift_type == 'all':
                if 'current_player' in message_concatenator.__name__:
                    warnings.warn('message_concatenator='+message_concatenator.__name__+' and label_shift_type='+str(label_shift_type)+' may be inconsistent')

                labels = chat_df['label'].tolist()
                chat_df['label'] = labels[label_shift_n:] + [None]*label_shift_n
                # remove shift_labels last messages of each match (predicting next toxicity is undefined)
                chat_df = chat_df[chat_df.groupby('match').cumcount(ascending=False) >= label_shift_n].reset_index()
            elif label_shift_type == 'current_player':
                if 'current_player' not in message_concatenator.__name__:
                    warnings.warn('message_concatenator='+message_concatenator.__name__+' and label_shift_type='+str(label_shift_type)+' may be inconsistent')
                # regroup chat
                to_regroup = []
                for match, match_df in chat_df.groupby('match'):
                    for player, player_match_df in match_df.groupby('player'):
                        # shift labels
                        labels = player_match_df['label'].to_list()
                        player_match_df['label'] = labels[label_shift_n:] + [None]*label_shift_n
                        # remove shift_labels last messages of each match (predicting next toxicity is undefined)
                        player_match_df = player_match_df[player_match_df.groupby('match').cumcount(ascending=False) >= label_shift_n]    
                        to_regroup.append(player_match_df)
                chat_df = pd.concat(to_regroup, axis=0).reset_index()

            else:
                raise NotImplementedError

        super().__init__(dataset_name=None, message_concatenator=message_concatenator, chat_df=chat_df, tokenizer=tokenizer)  
        self.format_label = format_label
        self.label_shift_type = label_shift_type
        self.label_shift_n = label_shift_n
        self.cache = False
        if cache:
            self.cache = self[:]

    def _getitem(self, index):
        if self.cache:
            return self.cache[index]
        else:
            return super()._getitem(index), self.format_label(self.chat_df['label'][index])

    def get_chat_label_user_id(self, index):
        return super()._getitem(index), self.format_label(self.chat_df['label'][index]), self.chat_df['user_id'][index]
        
    def get_balanced_subsamples(self, all_items=None):
        if not all_items:
            all_items = self[:]
        sorted_label_counts = sorted(Counter(list(zip(*all_items))[1]).items(), key=lambda x: x[1])
        # majority class // minority class
        majority_samples = [e for e in all_items if e[1]==sorted_label_counts[1][0]]
        minority_samples = [e for e in all_items if e[1]==sorted_label_counts[0][0]]
        random.shuffle(majority_samples)
        return [minority_samples + majority_samples[i:i+len(minority_samples)] for i in range(0, len(majority_samples), len(minority_samples))]


class UserMatchHistoryDataset(TorchDatasetGetItemWrapper):
    def __init__(self, message_concatenator, dataset_name='devinanzelmo_unlabeled_matches', chat_df=None, user_ids='all', default_history=None) -> None:
        super().__init__()
        self.default_history = default_history
        if isinstance(chat_df, pd.DataFrame) or chat_df!=None:
            self.chat_df = chat_df
        else:
            self.chat_df = pd.read_csv('/media/storage/adrien/toxicity-detection/'+dataset_name+'.csv').dropna(subset=['message', 'match', 'player', 'user_id']).drop_duplicates(['match', 'message', 'time', 'player', 'user_id']).fillna(np.nan).replace([np.nan], [None]).reset_index()
        if user_ids != 'all':
            assert type(user_ids) != str
            user_ids = set(user_ids)
            self.chat_df = self.chat_df[self.chat_df['user_id'].apply(lambda x: x in user_ids)]
        self.message_concatenator = message_concatenator
        self.user_hist_map = dict()
        for user_id, user_chat_df in self.chat_df.groupby('user_id'):
            self.user_hist_map[user_id] = [
                message_concatenator(user_match_chat_df['message'].to_list(), user_match_chat_df['player'].to_list()) for match, user_match_chat_df in user_chat_df.groupby('match')
            ]

    def __len__(self):
        return len(self.user_hist_map)
    
    def _getitem(self, user_id):
        if user_id in self.user_hist_map:
            return self.user_hist_map[user_id]
        else: # define what to do if user has no history
            return self.default_history