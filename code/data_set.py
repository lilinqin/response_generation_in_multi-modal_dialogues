
from torch.utils.data import Dataset, DataLoader
import PIL
import torch
from torchvision import datasets, models, transforms
from transformers import AutoTokenizer, BertTokenizer, BartTokenizer
import argparse
from tqdm import tqdm
import json
import copy
import os
from pysubs2 import SSAFile
import numpy as np

datasets_dir = '/home/lqli/video_dialog_generation/data'

type2season = {'train':list(range(1,9)), 'eval':[9], 'test':[10]}
seasons_length = [24,24,25,24,24,25,24,24,23,17]

class Utterance:
    def __init__(self, text, video=None, audio=None):
        self.text = text
        self.video = video
        self.audio = audio

def pad(sentences, max_len, num):
    for sentence in sentences:
        sentence += [num] * (max_len - len(sentence))
    return sentences

def get_utteraces_from_subtile(season,episode): # extract the subtitle time fragment from subtitle file
    read_path = os.path.join(datasets_dir,'subtitle','season'+str(season),'applied','friends.s'+str(season).zfill(2)+'e'+str(episode).zfill(2)+'.srt')
    subs = SSAFile.load(read_path,encoding='ISO-8859-1',format='srt')
    return subs

def get_data(seasons, max_turns=4):
    datas = []

    for ses in seasons:
        for epi in range(seasons_length[ses-1]):
            epi += 1
            subs = get_utteraces_from_subtile(ses,epi)
            his = []
            for segid, sub in enumerate(subs):
                segid = str(segid+1).zfill(3)
                text = sub.text.replace(r'\N',' ').replace(r'- ', ' ')
                text = text.replace(r'{\i0}', '').replace(r'{\i1}', '')
                text = ' '.join(text.split())
                video = os.path.join(datasets_dir, 'friendsVideoSegFeature', 'friends.s'+str(ses).zfill(2)+'e'+str(epi).zfill(2), 'seg'+str(segid).zfill(3)+'.npy')
                # video = os.path.join(datasets_dir, 'friendsVideoSegFeaturer21d', 'friends.s'+str(ses).zfill(2)+'e'+str(epi).zfill(2), 'seg'+str(segid).zfill(3)+'.npy')
                audio = os.path.join(datasets_dir, 'friendsAudioSegFeature', 'friends.s'+str(ses).zfill(2)+'e'+str(epi).zfill(2), 'seg'+str(segid).zfill(3)+'.npy')

                utterance = Utterance(text, video, audio)
                if segid != '001':
                    datas.append((his[:], utterance))
   
                his.append(utterance)
                if len(his) > max_turns * 2 + 1:
                    his = his[1:]

    return datas


def get_utteraces(seasons, max_turns=4):
    datas = []

    for ses in seasons:
        for epi in range(seasons_length[ses-1]):
            epi += 1
            subs = get_utteraces_from_subtile(ses,epi)
            his = None
            for segid, sub in enumerate(subs):
                segid = str(segid+1).zfill(3)
                text = sub.text.replace(r'\N',' ').replace(r'- ', ' ')
                text = text.replace(r'{\i0}', '').replace(r'{\i1}', '')
                text = ' '.join(text.split())
                video = os.path.join(datasets_dir, 'friendsVideoSegFeature', 'friends.s'+str(ses).zfill(2)+'e'+str(epi).zfill(2), 'seg'+str(segid).zfill(3)+'.npy')
                # video = os.path.join(datasets_dir, 'friendsVideoSegFeaturer21d', 'friends.s'+str(ses).zfill(2)+'e'+str(epi).zfill(2), 'seg'+str(segid).zfill(3)+'.npy')
                audio = os.path.join(datasets_dir, 'friendsAudioSegFeature', 'friends.s'+str(ses).zfill(2)+'e'+str(epi).zfill(2), 'seg'+str(segid).zfill(3)+'.npy')
                utterance = Utterance(text, video, audio)
                if segid == '001':
                    his = utterance
                else:
                    datas.append((his, utterance))
                    his = utterance
                
    return datas


class DialogDataset(Dataset):
    def __init__(self, args, data_type='train'):
        self.tokenizer = args.tokenizer
        self.mode = args.mode
        self.two_roles = args.two_roles
        self.max_turns = args.max_turns
        self.device = args.device

        seasons = type2season[data_type]
        self.datas = get_data(seasons, args.max_turns)

        self.max_len = args.max_len
        self.target_max_len = args.target_max_len
        self.video_id = self.tokenizer.encode('<video>', add_special_tokens=False)[0]
        self.audio_id = self.tokenizer.encode('<audio>', add_special_tokens=False)[0]


    def __getitem__(self, index):
        data = self.datas[index]
        his = data[0]
        ans = data[1]

        input_ids, attention_masks, token_type_ids, role_type_ids, video_features, audio_features, labels = self.build_input_from_segments(his, ans) 

        video_idx = [i for i,v in enumerate(input_ids) if v == self.video_id]
        audio_idx = [i for i,v in enumerate(input_ids) if v == self.audio_id]
   
        return input_ids, attention_masks, token_type_ids, role_type_ids, video_features, video_idx, audio_features, audio_idx, labels

    
    def build_input_from_segments(self, his, ans):
        cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id

        input_ids = []
        attention_masks = []
        token_type_ids = [] # text:0, video:1, audio:2
        role_type_ids = [] # 0, 
        video_features = []
        audio_features = []
        his_len = len(his)
        role_type = 0

        for i in range(his_len):
            idx = his_len - i - 1
            text = his[idx].text
            video = his[idx].video
            audio = his[idx].audio

            if role_type == 0:
                if 'Audio' in self.mode:
                    audio = np.load(audio).tolist()
                    # audio = np.load(audio)
                    # print(audio.shape)
                    # audio = audio.tolist()
                    temp_input_ids = [self.audio_id] * len(audio)
                    temp_input_ids = temp_input_ids + [sep_id]

                    temp_token_type_ids = [2] * len(temp_input_ids) # 2 represents audio

                    if len(temp_token_type_ids) + len(token_type_ids) > self.max_len - 1:
                        break
                    else:
                        role_type_ids = [role_type] * len(temp_token_type_ids) + role_type_ids
                        token_type_ids = temp_token_type_ids + token_type_ids
                        input_ids = temp_input_ids + input_ids
                        attention_masks = [1] * len(temp_token_type_ids) + attention_masks
                        audio_features.append(audio)

                if 'Video' in self.mode:
                    video = np.load(video).tolist()

                    # video = np.load(video)
                    # print(video.shape)
                    # video = video.tolist()
                    temp_input_ids = [self.video_id] * len(video)
                    temp_input_ids = temp_input_ids + [sep_id]

                    temp_token_type_ids = [1] * len(temp_input_ids) # 1 represents video

                    if len(temp_token_type_ids) + len(token_type_ids) > self.max_len - 1:
                        break
                    else:
                        role_type_ids = [role_type] * len(temp_token_type_ids) + role_type_ids
                        token_type_ids = temp_token_type_ids + token_type_ids
                        input_ids = temp_input_ids + input_ids
                        attention_masks = [1] * len(temp_token_type_ids) + attention_masks
                        video_features.extend(video)
                                 
                if 'Text' in self.mode:
                    temp_input_ids = self.tokenizer.encode(text, add_special_tokens=False)
                    temp_input_ids = temp_input_ids + [sep_id]

                    temp_token_type_ids = [0] * len(temp_input_ids) # 0 represents text

                    if len(temp_token_type_ids) + len(token_type_ids) > self.max_len - 1:
                        break
                    else:
                        role_type_ids = [role_type] * len(temp_token_type_ids) + role_type_ids
                        token_type_ids = temp_token_type_ids + token_type_ids
                        input_ids = temp_input_ids + input_ids
                        attention_masks = [1] * len(temp_token_type_ids) + attention_masks

            elif role_type == 1:
                if self.two_roles:
                    if 'Audio' in self.mode:
                        audio = np.load(audio).tolist()
                        temp_input_ids = [self.audio_id] * len(audio)
                        temp_input_ids = temp_input_ids + [sep_id]

                        temp_token_type_ids = [2] * len(temp_input_ids) # 2 represents audio

                        if len(temp_token_type_ids) + len(token_type_ids) > self.max_len - 1:
                            break
                        else:
                            role_type_ids = [role_type] * len(temp_token_type_ids) + role_type_ids
                            token_type_ids = temp_token_type_ids + token_type_ids
                            input_ids = temp_input_ids + input_ids
                            attention_masks = [1] * len(temp_token_type_ids) + attention_masks
                            audio_features.append(audio)
                
                    if 'Video' in self.mode:
                        video = np.load(video).tolist()
                        temp_input_ids = [self.video_id] * len(video)
                        temp_input_ids = temp_input_ids + [sep_id]

                        temp_token_type_ids = [1] * len(temp_input_ids) # 1 represents video

                        if len(temp_token_type_ids) + len(token_type_ids) > self.max_len - 1:
                            break
                        else:
                            role_type_ids = [role_type] * len(temp_token_type_ids) + role_type_ids
                            token_type_ids = temp_token_type_ids + token_type_ids
                            input_ids = temp_input_ids + input_ids
                            attention_masks = [1] * len(temp_token_type_ids) + attention_masks
                            video_features.extend(video)
                            
                    if 'Text' in self.mode:
                        temp_input_ids = self.tokenizer.encode(text, add_special_tokens=False)
                        temp_input_ids = temp_input_ids + [sep_id]

                        temp_token_type_ids = [0] * len(temp_input_ids) # 0 represents text

                        if len(temp_token_type_ids) + len(token_type_ids) > self.max_len - 1:
                            break
                        else:
                            role_type_ids = [role_type] * len(temp_token_type_ids) + role_type_ids
                            token_type_ids = temp_token_type_ids + token_type_ids
                            input_ids = temp_input_ids + input_ids
                            attention_masks = [1] * len(temp_token_type_ids) + attention_masks
                else:
                    temp_input_ids = self.tokenizer.encode(text, add_special_tokens=False)
                    temp_input_ids = temp_input_ids + [sep_id]

                    temp_token_type_ids = [0] * len(temp_input_ids) # 0 represents text

                    if len(temp_token_type_ids) + len(token_type_ids) > self.max_len - 1:
                        break
                    else:
                        role_type_ids = [role_type] * len(temp_token_type_ids) + role_type_ids
                        token_type_ids = temp_token_type_ids + token_type_ids
                        input_ids = temp_input_ids + input_ids
                        attention_masks = [1] * len(temp_token_type_ids) + attention_masks

            role_type = (role_type + 1) % 2
        
        token_type_ids = token_type_ids[0:1] + token_type_ids
        role_type_ids = role_type_ids[0:1] + role_type_ids
        input_ids = [cls_id] + input_ids
        attention_masks = [1] + attention_masks

        video_features.reverse()
        audio_features.reverse()

        labels = self.tokenizer.encode(ans.text, add_special_tokens=True, max_length=self.target_max_len, truncation=True)

        return input_ids, attention_masks, token_type_ids, role_type_ids, video_features, audio_features, labels


    def collate_fn(self, batch):
        input_ids, attention_masks, token_type_ids, role_type_ids, video_features, video_idx, audio_features, audio_idx, labels = list(zip(*batch))
        pad_id = self.tokenizer.pad_token_id

        max_len = max((len(x) for x in input_ids))
        input_ids = pad(input_ids, max_len, pad_id)
        attention_masks = pad(attention_masks, max_len, 0)
        token_type_ids = pad(token_type_ids, max_len, 0)
        role_type_ids = pad(role_type_ids, max_len, 0)

        max_label_len = max((len(x) for x in labels))
        labels = pad(labels, max_label_len, pad_id)

        max_video_len = max((len(x) for x in video_idx))
        video_idx = pad(video_idx, max_video_len, -1)

        max_audio_len = max((len(x) for x in audio_idx))
        audio_idx = pad(audio_idx, max_audio_len, -1)

        input_ids = torch.LongTensor(input_ids).to(self.device)
        attention_masks = torch.Tensor(attention_masks).to(self.device)
        token_type_ids = torch.LongTensor(token_type_ids).to(self.device)
        role_type_ids = torch.LongTensor(role_type_ids).to(self.device)
        
        video_temp = []
        for item in video_features:
            for seg in item:
                video_temp.append(seg)
        if video_temp:
            video_features = torch.Tensor(video_temp).to(self.device)
        else:
            video_features = None
        
        audio_temp = []
        for item in audio_features:
            for segs in item:
                for seg in segs:
                    audio_temp.append(seg)
        if audio_temp:
            audio_features = torch.Tensor(audio_temp).to(self.device)
        else:
            audio_features = None
        
        labels = torch.LongTensor(labels).to(self.device)
        
        if self.mode in ['Text', 'Video', 'Audio', 'None']:
            token_type_ids = None
        if self.mode in ['None']:
            role_type_ids = None

        output = {
            'input_ids':input_ids,
            'attention_mask':attention_masks,
            'role_type':role_type_ids,
            'token_type':token_type_ids,
            'video_features':video_features,
            'video_idx':video_idx,
            'audio_features':audio_features,
            'audio_idx':audio_idx,
            'labels':labels
        }
        
        return output

    def __len__(self):
        # return 2
        # return 10000
        return len(self.datas)
        


class DialogContrastWithGenerationDataset(Dataset):
    def __init__(self, args, data_type='train'):
        self.tokenizer = args.tokenizer
        # self.mode = args.mode
        self.mode = 'TextAudio'
        self.two_roles = args.two_roles
        self.max_turns = args.max_turns
        self.device = args.device

        seasons = type2season[data_type]
        self.datas = get_utteraces(seasons, args.max_turns)

        self.max_len = args.max_len
        self.target_max_len = args.target_max_len
        self.video_id = self.tokenizer.encode('<video>', add_special_tokens=False)[0]
        self.audio_id = self.tokenizer.encode('<audio>', add_special_tokens=False)[0]


    def __getitem__(self, index):
        utterance, ans = self.datas[index]
        
        text_input_ids, text_attention_masks, text_token_type_ids, video_input_ids, video_attention_masks, video_token_type_ids, audio_input_ids, audio_attention_masks, audio_token_type_ids, video_features, audio_features, labels = self.build_input_from_segments(utterance, ans) 

        video_idx = [i for i,v in enumerate(video_input_ids) if v == self.video_id and i != 0]
        audio_idx = [i for i,v in enumerate(audio_input_ids) if v == self.audio_id and i != 0]
 
        return text_input_ids, text_attention_masks, text_token_type_ids, video_input_ids, video_attention_masks, video_token_type_ids, audio_input_ids, audio_attention_masks, audio_token_type_ids, video_features, audio_features, video_idx, audio_idx, labels
    

    def build_input_from_segments(self, utterace, ans):
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        text_attention_masks = []
        text_token_type_ids = [] # text:0, video:1, audio:2

        video_features = []
        audio_features = []

        text = utterace.text
        text_input_ids = self.tokenizer.encode(text)
        text_attention_masks = [1] * len(text_input_ids)
        text_token_type_ids = [0] * len(text_input_ids)

        video = utterace.video
        video = np.load(video).tolist()
        video_input_ids = [bos_token_id] + [self.video_id] * len(video) + [eos_token_id]
        video_attention_masks = [1] * len(video_input_ids)
        video_token_type_ids = [1] * len(video_input_ids) # 1 represents video
        video_features.extend(video)

        audio = utterace.audio
        audio = np.load(audio).tolist()
        audio_input_ids = [bos_token_id] + [self.audio_id] * len(audio) + [eos_token_id]
        audio_attention_masks = [1] * len(audio_input_ids)
        audio_token_type_ids = [2] * len(audio_input_ids) # 2 represents audio
        audio_features.append(audio)

        labels = self.tokenizer.encode(ans.text, add_special_tokens=True, max_length=self.target_max_len, truncation=True)

        return text_input_ids, text_attention_masks, text_token_type_ids, video_input_ids, video_attention_masks, video_token_type_ids, audio_input_ids, audio_attention_masks, audio_token_type_ids, video_features, audio_features, labels


    def collate_fn(self, batch):
        text_input_ids, text_attention_masks, text_token_type_ids, video_input_ids, video_attention_masks, video_token_type_ids, audio_input_ids, audio_attention_masks, audio_token_type_ids, video_features, audio_features, video_idx, audio_idx, labels = list(zip(*batch))
        pad_id = self.tokenizer.pad_token_id
        # print(video_features)

        max_len = max((len(x) for x in text_input_ids))
        text_input_ids = pad(text_input_ids, max_len, pad_id)
        text_attention_masks = pad(text_attention_masks, max_len, 0)
        text_token_type_ids = pad(text_token_type_ids, max_len, 0)

        max_len = max((len(x) for x in video_input_ids))
        video_input_ids = pad(video_input_ids, max_len, pad_id)
        video_attention_masks = pad(video_attention_masks, max_len, 0)
        video_token_type_ids = pad(video_token_type_ids, max_len, 0)

        max_len = max((len(x) for x in audio_input_ids))
        audio_input_ids = pad(audio_input_ids, max_len, pad_id)
        audio_attention_masks = pad(audio_attention_masks, max_len, 0)
        audio_token_type_ids = pad(audio_token_type_ids, max_len, 0)
        

        max_video_len = max((len(x) for x in video_idx))
        video_idx = pad(video_idx, max_video_len, -1)

        max_audio_len = max((len(x) for x in audio_idx))
        audio_idx = pad(audio_idx, max_audio_len, -1)


        text_input_ids = torch.LongTensor(text_input_ids).to(self.device)
        text_attention_masks = torch.Tensor(text_attention_masks).to(self.device)
        text_token_type_ids = torch.LongTensor(text_token_type_ids).to(self.device)

        video_input_ids = torch.LongTensor(video_input_ids).to(self.device)
        video_attention_masks = torch.Tensor(video_attention_masks).to(self.device)
        video_token_type_ids = torch.LongTensor(video_token_type_ids).to(self.device)

        audio_input_ids = torch.LongTensor(audio_input_ids).to(self.device)
        audio_attention_masks = torch.Tensor(audio_attention_masks).to(self.device)
        audio_token_type_ids = torch.LongTensor(audio_token_type_ids).to(self.device)


        video_temp = []
        for item in video_features:
            for seg in item:
                video_temp.append(seg)
        if video_temp:
            video_features = torch.Tensor(video_temp).to(self.device)
        else:
            video_features = None
        
        audio_temp = []
        for item in audio_features:
            for segs in item:
                for seg in segs:
                    audio_temp.append(seg)
        if audio_temp:
            audio_features = torch.Tensor(audio_temp).to(self.device)
        else:
            audio_features = None
        

        max_label_len = max((len(x) for x in labels))
        labels = pad(labels, max_label_len, pad_id)
        labels = torch.LongTensor(labels).to(self.device)
        # labels = None

        output = {}

        output['text_input'] = {
            'input_ids':text_input_ids,
            'attention_mask':text_attention_masks,
            'role_type':None,
            'token_type':text_token_type_ids,
            'video_features':None,
            'video_idx':None,
            'audio_features':None,
            'audio_idx':None,
            'labels':labels
        }
        if 'Video' in self.mode:
            output['video_input'] = {
                'input_ids':video_input_ids,
                'attention_mask':video_attention_masks,
                'role_type':None,
                'token_type':video_token_type_ids,
                'video_features':video_features,
                'video_idx':video_idx,
                'audio_features':None,
                'audio_idx':None,
                'labels':labels
            }
        else:
            output['video_input'] = None
        
        if 'Audio' in self.mode:
            output['audio_input'] = {
                'input_ids':audio_input_ids,
                'attention_mask':audio_attention_masks,
                'role_type':None,
                'token_type':audio_token_type_ids,
                'video_features':None,
                'video_idx':None,
                'audio_features':audio_features,
                'audio_idx':audio_idx,
                'labels':labels
            }
        else:
            output['audio_input'] = None

        return output

    def __len__(self):
        # return 2
        # return 10000
        return len(self.datas)


# class PromptDataset(Dataset):
#     def __init__(self, args, data_type='train'):
#         self.tokenizer = args.tokenizer
#         self.max_turns = args.max_turns
#         self.device = args.device

#         seasons = type2season[data_type]
#         self.datas = get_data(seasons, args.max_turns)

#         self.max_len = args.max_len
#         self.target_max_len = args.target_max_len


#     def __getitem__(self, index):
#         data = self.datas[index]
#         his = data[0]
#         ans = data[1]

#         input_ids, attention_masks, token_type_ids, role_type_ids, labels, query, query_attention_mask = self.build_input_from_segments(his, ans) 

#         query_idx = [i for i,v in enumerate(token_type_ids) if v == 1]
   
#         return input_ids, attention_masks, token_type_ids, role_type_ids, labels, query, query_attention_mask, query_idx

    

#     def build_input_from_segments(self, his, ans):
#         cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id

#         input_ids = []
#         attention_masks = []
#         token_type_ids = [] # text:0, video:1, audio:2
#         role_type_ids = [] # 0, 

#         labels = []

#         his_len = len(his)

#         role_type = 0
#         # query
#         query = his[his_len-1].text
#         query = self.tokenizer.encode(query, add_special_tokens=True)
#         temp_token_type_ids = [1] * len(query) # 3 represents prompt

#         role_type_ids = [role_type] * len(temp_token_type_ids) + role_type_ids
#         token_type_ids = temp_token_type_ids + token_type_ids
#         input_ids = query + input_ids
#         query_attention_mask = [1] * len(temp_token_type_ids) 
#         attention_masks = query_attention_mask + attention_masks
#         query_len = len(query)


#         for i in range(his_len):
#             idx = his_len - i - 1
#             text = his[idx].text

#             temp_input_ids = self.tokenizer.encode(text, add_special_tokens=False)
#             temp_input_ids = temp_input_ids + [sep_id]

#             temp_token_type_ids = [0] * len(temp_input_ids) # 0 represents text

#             if len(temp_token_type_ids) + len(token_type_ids) > self.max_len - 1 - query_len:
#                 break
#             else:
#                 role_type_ids = [role_type] * len(temp_token_type_ids) + role_type_ids
#                 token_type_ids = temp_token_type_ids + token_type_ids
#                 input_ids = temp_input_ids + input_ids
#                 attention_masks = [1] * len(temp_token_type_ids) + attention_masks
            

#             role_type = (role_type + 1) % 2
        
#         token_type_ids = token_type_ids[0:1] + token_type_ids
#         role_type_ids = role_type_ids[0:1] + role_type_ids
#         input_ids = [cls_id] + input_ids
#         attention_masks = [1] + attention_masks


#         labels = self.tokenizer.encode(ans.text, add_special_tokens=True, max_length=self.target_max_len, truncation=True)

#         return input_ids, attention_masks, token_type_ids, role_type_ids, labels, query, query_attention_mask


#     def collate_fn(self, batch):
#         input_ids, attention_masks, token_type_ids, role_type_ids, labels, querys, query_attention_masks, query_idx = list(zip(*batch))
#         pad_id = self.tokenizer.pad_token_id
#         # print(video_features)

#         max_len = max((len(x) for x in input_ids))
#         input_ids = pad(input_ids, max_len, pad_id)
#         attention_masks = pad(attention_masks, max_len, 0)
#         token_type_ids = pad(token_type_ids, max_len, 0)
#         role_type_ids = pad(role_type_ids, max_len, 0)

#         max_len = max((len(x) for x in input_ids))
#         querys = pad(querys, max_len, pad_id)
#         query_attention_masks = pad(query_attention_masks, max_len, 0)


#         max_label_len = max((len(x) for x in labels))
#         labels = pad(labels, max_label_len, pad_id)


#         max_query_len = max((len(x) for x in query_idx))
#         query_idx = pad(query_idx, max_query_len, -1)

#         input_ids = torch.LongTensor(input_ids).to(self.device)
#         attention_masks = torch.Tensor(attention_masks).to(self.device)
#         token_type_ids = torch.LongTensor(token_type_ids).to(self.device)
#         role_type_ids = torch.LongTensor(role_type_ids).to(self.device)

#         querys = torch.LongTensor(querys).to(self.device)
#         query_attention_masks = torch.Tensor(query_attention_masks).to(self.device)
        
#         labels = torch.LongTensor(labels).to(self.device)
        
#         output = {
#             'input_ids':input_ids,
#             'attention_mask':attention_masks,
#             'role_type':role_type_ids,
#             'token_type':token_type_ids,
#             'querys':querys,
#             'query_attention_masks':query_attention_masks,
#             'query_idx':query_idx,
#             'labels':labels
#         }
 
#         return output

#     def __len__(self):
#         return 2
#         # return 10000
#         return len(self.datas)


# class AdapBridgeDataset(Dataset):
#     def __init__(self, args, data_type='train'):
#         self.tokenizer = args.tokenizer
#         self.max_turns = args.max_turns
#         self.device = args.device

#         seasons = type2season[data_type]
#         self.datas = get_data(seasons, args.max_turns)

#         self.max_len = args.max_len
#         self.target_max_len = args.target_max_len


#     def __getitem__(self, index):
#         data = self.datas[index]
#         his = data[0]
#         ans = data[1]

#         input_ids, attention_masks, token_type_ids, role_type_ids, labels, = self.build_input_from_segments(his, ans) 

#         return input_ids, attention_masks, token_type_ids, role_type_ids, labels

    

#     def build_input_from_segments(self, his, ans):
#         cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id

#         input_ids = []
#         attention_masks = []
#         token_type_ids = [] # text:0, video:1, audio:2
#         role_type_ids = [] # 0, 

#         labels = []

#         his_len = len(his)

#         role_type = 0
        

#         for i in range(his_len):
#             idx = his_len - i - 1
#             text = his[idx].text

#             temp_input_ids = self.tokenizer.encode(text, add_special_tokens=False)
#             temp_input_ids = temp_input_ids + [sep_id]

#             temp_token_type_ids = [0] * len(temp_input_ids) # 0 represents text

#             if len(temp_token_type_ids) + len(token_type_ids) > self.max_len - 1:
#                 break
#             else:
#                 role_type_ids = [role_type] * len(temp_token_type_ids) + role_type_ids
#                 token_type_ids = temp_token_type_ids + token_type_ids
#                 input_ids = temp_input_ids + input_ids
#                 attention_masks = [1] * len(temp_token_type_ids) + attention_masks
            

#             role_type = (role_type + 1) % 2
        
#         token_type_ids = token_type_ids[0:1] + token_type_ids
#         role_type_ids = role_type_ids[0:1] + role_type_ids
#         input_ids = [cls_id] + input_ids
#         attention_masks = [1] + attention_masks


#         labels = self.tokenizer.encode(ans.text, add_special_tokens=True, max_length=self.target_max_len, truncation=True)

#         return input_ids, attention_masks, token_type_ids, role_type_ids, labels


#     def collate_fn(self, batch):
#         input_ids, attention_masks, token_type_ids, role_type_ids, labels = list(zip(*batch))
#         pad_id = self.tokenizer.pad_token_id
#         # print(video_features)

#         max_len = max((len(x) for x in input_ids))
#         input_ids = pad(input_ids, max_len, pad_id)
#         attention_masks = pad(attention_masks, max_len, 0)
#         role_type_ids = pad(role_type_ids, max_len, 0)

#         max_label_len = max((len(x) for x in labels))
#         labels = pad(labels, max_label_len, pad_id)


#         input_ids = torch.LongTensor(input_ids).to(self.device)
#         attention_masks = torch.Tensor(attention_masks).to(self.device)
#         role_type_ids = torch.LongTensor(role_type_ids).to(self.device)

        
#         labels = torch.LongTensor(labels).to(self.device)
        
#         output = {
#             'input_ids':input_ids,
#             'attention_mask':attention_masks,
#             'role_type':role_type_ids,
#             'labels':labels
#         }
 
#         return output

#     def __len__(self):
#         # return 2
#         # return 10000
#         return len(self.datas)



# class ProphetChatDataset(Dataset):
#     def __init__(self, args, responses=None, futures=None, data_type='train'):
#         self.tokenizer = args.tokenizer
#         self.max_turns = args.max_turns
#         self.device = args.device

#         seasons = type2season[data_type]
#         self.datas = get_data(seasons, args.max_turns)
#         self.responses = responses
#         self.futures = futures

#         self.max_len = args.max_len
#         self.target_max_len = args.target_max_len


#     def __getitem__(self, index):
#         data = self.datas[index]
#         his = data[0]
#         ans = data[1]
#         future = self.futures[index] if self.futures else None
#         response = self.responses[index] if self.responses else None

#         input_ids, attention_masks, token_type_ids, role_type_ids, labels, = self.build_input_from_segments(his, ans, response, future) 

#         return input_ids, attention_masks, token_type_ids, role_type_ids, labels

    

#     def build_input_from_segments(self, his, ans, response=None, future=None):
#         cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id

#         input_ids = []
#         attention_masks = []
#         token_type_ids = [] # text:0, video:1, audio:2
#         role_type_ids = [] # 0, 

#         labels = []

#         his_len = len(his)

#         role_type = 0

#         if response is not None:
#             temp_input_ids = self.tokenizer.encode(response, add_special_tokens=False)
#             temp_input_ids = temp_input_ids + [sep_id]

#             temp_token_type_ids = [0] * len(temp_input_ids) # 0 represents text
            
#             role_type_ids = [role_type] * len(temp_token_type_ids) + role_type_ids
#             token_type_ids = temp_token_type_ids + token_type_ids
#             input_ids = temp_input_ids + input_ids
#             attention_masks = [1] * len(temp_token_type_ids) + attention_masks
            
#             role_type = (role_type + 1) % 2
#             # print('response')
        
#         if future is not None:
#             temp_input_ids = self.tokenizer.encode(future, add_special_tokens=False)
#             temp_input_ids = temp_input_ids + [sep_id]

#             temp_token_type_ids = [0] * len(temp_input_ids) # 0 represents text
            
#             role_type_ids = [role_type] * len(temp_token_type_ids) + role_type_ids
#             token_type_ids = temp_token_type_ids + token_type_ids
#             input_ids = temp_input_ids + input_ids
#             attention_masks = [1] * len(temp_token_type_ids) + attention_masks
            
#             # role_type = (role_type + 1) % 2
#             # print('future')
        

#         for i in range(his_len):
#             idx = his_len - i - 1
#             text = his[idx].text

#             temp_input_ids = self.tokenizer.encode(text, add_special_tokens=False)
#             temp_input_ids = temp_input_ids + [sep_id]

#             temp_token_type_ids = [0] * len(temp_input_ids) # 0 represents text

#             if len(temp_token_type_ids) + len(token_type_ids) > self.max_len - 1:
#                 break
#             else:
#                 role_type_ids = [role_type] * len(temp_token_type_ids) + role_type_ids
#                 token_type_ids = temp_token_type_ids + token_type_ids
#                 input_ids = temp_input_ids + input_ids
#                 attention_masks = [1] * len(temp_token_type_ids) + attention_masks
            

#             role_type = (role_type + 1) % 2
        
#         token_type_ids = token_type_ids[0:1] + token_type_ids
#         role_type_ids = role_type_ids[0:1] + role_type_ids
#         input_ids = [cls_id] + input_ids
#         attention_masks = [1] + attention_masks


#         labels = self.tokenizer.encode(ans.text, add_special_tokens=True, max_length=self.target_max_len, truncation=True)

#         return input_ids, attention_masks, token_type_ids, role_type_ids, labels


#     def collate_fn(self, batch):
#         input_ids, attention_masks, token_type_ids, role_type_ids, labels = list(zip(*batch))
#         pad_id = self.tokenizer.pad_token_id
#         # print(video_features)

#         max_len = max((len(x) for x in input_ids))
#         input_ids = pad(input_ids, max_len, pad_id)
#         attention_masks = pad(attention_masks, max_len, 0)
#         role_type_ids = pad(role_type_ids, max_len, 0)

#         max_label_len = max((len(x) for x in labels))
#         labels = pad(labels, max_label_len, pad_id)


#         input_ids = torch.LongTensor(input_ids).to(self.device)
#         attention_masks = torch.Tensor(attention_masks).to(self.device)
#         role_type_ids = torch.LongTensor(role_type_ids).to(self.device)

        
#         labels = torch.LongTensor(labels).to(self.device)
        
#         output = {
#             'input_ids':input_ids,
#             'attention_mask':attention_masks,
#             'role_type':role_type_ids,
#             'labels':labels
#         }
 
#         return output

#     def __len__(self):
#         return 2
#         # return 10000
#         return len(self.datas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', default=512, type=int)
    parser.add_argument('--target_max_len', default=256, type=int)
    parser.add_argument('--mode', default='TextVideoAudio', type=str)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', default=device, type=str, help='the device gpu or cpu')
    parser.add_argument('--max_turns', default=4, type=int, help='max_turns')

    args = parser.parse_args()

    args.tokenizer = BartTokenizer.from_pretrained('/home/llq/project/video_dialog_generation/pretrain/bart-base')
    args.tokenizer.add_tokens(["<video>"], special_tokens=True)
    args.tokenizer.add_tokens(["<audio>"], special_tokens=True)
    args.two_roles = True
    dataset = PromptDataset(args, 'train')

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    print(len(dataloader))
    # print(len())
    for x in tqdm(dataloader):
        print(x['input_ids'])
        break
        1
        
    print('ok')