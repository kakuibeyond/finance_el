import imp
import torch
from torch.utils.data import Dataset
from config import DATASET_FILENAME,DATASET_BASE_DIR, MAX_DESC_LEN,MAX_SEQ_LEN,PRETRAIN_MODEL_NAME
import os
from utils.io import format_filename, pickle_load
import copy
from random import sample
from transformers import BertTokenizer

def load_data(data_type):
    if data_type in DATASET_FILENAME:
        data = pickle_load(format_filename(DATASET_BASE_DIR, DATASET_FILENAME[data_type]))
    else:
        raise ValueError('data tye not understood: {}'.format(data_type))
    return data


class LinkingDataset(Dataset):
    def __init__(self,
                 file_mode,
                 tkn_type='bert',
                 entity_desc=None,
                 n_neg=1):
        self.file_mode = file_mode
        '''
        {
            'idx':idx,
            'text':item['text'],
            'entity_list':entity_list,
            'true_entity':true_entity
            }
        '''
        self.data = load_data(file_mode)
        # self.char_vocab = char_vocab # char2idx 将字符转换为idx，用于使用后面的百度词向量，BERT编码不需要这个环节，直接将cls+seq送入BERT即可
        self.entity_desc = entity_desc # 知识库实体描述信息 用来取实体描述文本 这里直接用文本全称代替desc
        # self.alias_data = alias_data # 相当于mention2entity_id 用来取候选实体
        # self.kb_ids = list(subject_data.keys())
        # self.max_desc_len = max_desc_len
        self.n_neg = n_neg # 控制负样本的比例 默认正负1：1


    def __getitem__(self, index):
        texts, descs, labels = [], [], []
        data_one = self.data[index]
        raw_text = data_one['text']
        # text_id = [self.char_vocab.get(c, 1) for c in raw_text] # 文本的id形式
        cand_entity = copy.deepcopy(data_one['entity_list']) # 所有候选实体名称 包括正确和错误
        true_entity = data_one['true_entity']
        while true_entity in cand_entity: # cand_ents可能未去重 有多个pos_ent
            cand_entity.remove(true_entity) # 深拷贝的list 不会影响原始列表的值
        
        # generate negative samples
        if len(cand_entity) == 0:
            # self.entity_desc.keys()里面就是一系列entity_id，从中取样
            neg_ents = sample(self.entity_desc.keys(), self.n_neg)
        elif len(cand_entity) < self.n_neg:
            neg_ents = cand_entity + sample(self.entity_desc.keys(), self.n_neg - len(cand_entity))
        else:
            neg_ents = sample(cand_entity, self.n_neg)

        # pos_desc_id = [self.char_vocab.get(c, 1) for c in true_entity]
        pos_desc = true_entity
        descs.append(pos_desc)
        texts=(self.n_neg + 1) * [raw_text]
        labels.append(1)
        for neg_ent in neg_ents:
            neg_desc = self.entity_desc[neg_ent] # 负样本文本
            # neg_desc_ids.append(neg_desc_id)
            # pos_desc_ids.append(pos_desc_id)# 正样本复制n_neg份
            descs.append(neg_desc)
            labels.append(0)

        # text_ids shape:[n_neg+1, seq_len] seq_len不定长 因此后面需要pad
        # desc_ids 同上
        # labels: 一维数组 n_neg+1
        return texts, descs, labels

    
    

def collate_fn_link(self, batch, tokenizer:BertTokenizer):
    batch_texts,batch_descs,batch_labels=[],[],[]
    for i, item in enumerate(batch):
        texts, descs, labels = item
        batch_texts.extend(texts) # batch_texts一维数组 每个元素为一句话:(batch*(neg+1), seq_len) 句子不定长
        batch_descs.extend(descs)
        batch_labels.extend(labels)
    batch_text_input=tokenizer(batch_texts,padding=True,max_length=MAX_SEQ_LEN,
                                truncation=True,return_tensors='pt')
    batch_desc_input=tokenizer(batch_descs,padding=True,max_length=MAX_DESC_LEN,
                                truncation=True,return_tensors='pt')
    batch_labels=torch.tensor(batch_labels)
    return {'text_ids':batch_text_input,
            'desc_ids':batch_desc_input,
            'labels':batch_labels}