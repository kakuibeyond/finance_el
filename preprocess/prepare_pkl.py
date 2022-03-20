# 根据标注好的语料csv 准备各方面训练数据 存入pkl文件
# 1.实体信息：mention2entity,entity2idx,idx2entity
# 2.语料信息：train_data,dev_data,test_data
# 3.训练得到的词向量及字典
#   char_vocab, idx2char，c2v_embeddings.npy
#   word_vocab, idx2word, w2v_embeddings.npy

import os
import sys
from tkinter.tix import Tree
curr_dir=os.path.dirname(__file__)
sys.path.append(os.path.join(curr_dir,'..'))
import pandas as pd
from collections import defaultdict
from utils.io import pickle_dump,format_filename, pickle_load
import pickle
import jieba
from config import *
from sklearn.model_selection import train_test_split
from data.anno_res0227 import process_anno as pa
from tqdm import tqdm
from utils.embedding import train_w2v
import numpy as np

# 从标注之前的3w+语料中获取所有实体
def get_all_entitys(m2e, before_anno_path):
    entity_list=set()
    entity2idx=defaultdict() # 在语料中出现过的实体，用唯一序号表示
    idx2entity=defaultdict()
    mention2entity = defaultdict(list) # 语料中出现过的 文本对文本
    # entity_to_mention = defaultdict(list) # 文本对文本
    # 实体名称到实体描述的字典 需要关注一下cndbpedia原始数据 只爬取下来语料中有的实体
    entity_desc = defaultdict() # 文本对文本
  
    df=pd.read_csv(before_anno_path,sep='\t',index_col=0)
    print('='*20+'开始读取所有实体'+'='*20)
    for _,row in df.iterrows():
        men=row['mention'].split('/')[0]
        candidate_ens = list(m2e[men].values())
        entity_list.update(candidate_ens) # 这里由于是用来标注的 所以m2e中一定存在且实体有效
        if men not in mention2entity:
            mention2entity[men]=candidate_ens
    print('='*20+'所有实体解析结束'+'='*20)
    print('实体数量：{}'.format(len(entity_list)))
    idx2entity={idx:ety for idx,ety in enumerate(entity_list)} # 0开始
    entity2idx={ety:idx for idx,ety in enumerate(entity_list)}

    return mention2entity,entity2idx,idx2entity


# 从标注数据里取出训练测试语料
def get_train_and_test_data(anno_path):
    print('='*20+'开始读取所有标注语料'+'='*20)
    df=pd.read_csv(anno_path,sep='\t',index_col=0)
    print('语料规模：{}'.format(df.shape))
    error_idx=[] # 存在标注错误的语料

    all_data=[]
    train_data=[]
    test_data=[]
    for idx,item in df.iterrows():
        mention=item['mention'].split('/')[0]
        entity_list=mention2entity[mention] # 文本
        anno=item['annotation']
        if anno>0:
            try:
                true_entity=m2e[mention][anno] # 对应的正确实体名称
                all_data.append({
                    'idx':idx,
                    'text':item['text'],
                    'entity_list':entity_list,
                    'true_entity':true_entity
                    })
            except Exception:
                print('mention2entity异常，idx:{},text:{},当前mention:{},已存标注列表:{},标注anno:{}'
                        .format(idx,item['text'],mention,m2e[mention],anno))
                error_idx.append(idx)
    if len(error_idx)>0:
        print('开始修正标注错误语料')
        pa.check_anno_list(df,error_idx)
        df.to_csv(anno_dir,sep='\t') # 默认会存储索引

    train_data,test_data=train_test_split(all_data,test_size=0.2)
    test_data,dev_data=train_test_split(test_data,test_size=0.5)
    
    print('训练数据{}条,第一条为：{}'.format(len(train_data),train_data[0]))
    print('验证数据{}条,第一条为：{}'.format(len(dev_data),dev_data[0]))
    print('测试数据{}条,第一条为：{}'.format(len(test_data),test_data[0]))

    print('='*20+'所有标注语料解析结束'+'='*20)
    return train_data,dev_data,test_data


# 根据实体描述信息和训练数据构建词典和语料
# 实体描述信息 这里先用idx2entity的value表示 即实体的全称
# char2idx(我：3), idx2char（3：我）, corpus（每一条desc/训练文本都作为语料的一句，每一条由单字列表组成）
# 注意：char2idx和idx2char都做了最低频次过滤，但corpus内部保留了所有原始信息
def load_char_vocab_and_corpus(entity_desc, train_data, min_count=2):
    chars = dict()
    corpus = []
    for desc in tqdm(iter(entity_desc.values())):
        for c in desc:
            chars[c] = chars.get(c, 0) + 1# 统计每个字符出现的次数，如"我：33"
        corpus.append(list(desc))#每一个句话都作为语料的一个元素，每句话由单字列表组成
    for data in tqdm(iter(train_data)):
        for c in data['text']:
            chars[c] = chars.get(c, 0) + 1#训练数据中的字也计入字出现频次
        corpus.append(list(data['text']))
    chars = {i: j for i, j in chars.items() if j >= min_count}#至少出现2次才保留
    idx2char = {i + 2: j for i, j in enumerate(chars)}  # 0: mask, 1: padding # 枚举的j为字典char的key，即字符
    char2idx = {j: i for i, j in idx2char.items()}
    return char2idx, idx2char, corpus


def load_word_vocab_and_corpus(entity_desc, train_data, min_count=2):
    words = dict()
    corpus = []
    for desc in tqdm(iter(entity_desc.values())):
        desc_cut = jieba.lcut(desc) #jieba默认分词，返回list
        for w in desc_cut:
            words[w] = words.get(w, 0) + 1 # 统计每个词词频
        corpus.append(desc_cut)
    for data in tqdm(iter(train_data)):
        text_cut = jieba.lcut(data['text'])
        for w in text_cut:
            words[w] = words.get(w, 0) + 1
        corpus.append(text_cut)
    words = {i: j for i, j in words.items() if j >= min_count}
    idx2word = {i + 2: j for i, j in enumerate(words)}  # 0: mask, 1: padding
    word2idx = {j: i for i, j in idx2word.items()}
    return word2idx, idx2word, corpus



if __name__ == '__main__':
    os.makedirs(PROCESSED_DATA_DIR,exist_ok=True)# 用于保留pkl文件的路径

    # 0301跑完一次 得到全集mention2entity,entity2idx,idx2entity并存储
    before_anno_path='data/seg_anno.csv'
    with open('data/m2e_all.pkl','rb') as fr2: # 字典key为mention，value为候选实体列表的字典
        m2e=pickle.load(fr2)
    mention2entity,entity2idx,idx2entity= get_all_entitys(m2e,before_anno_path)
    # pickle_dump(format_filename(PROCESSED_DATA_DIR,MENTION_TO_ENTITY_FILENAME), mention2entity)
    # pickle_dump(format_filename(PROCESSED_DATA_DIR,ENTITY_TO_IDX), entity2idx)
    # pickle_dump(format_filename(PROCESSED_DATA_DIR,IDX_TO_ENTITY), idx2entity)


    # 处理标注数据 得到训练测试语料
    anno_dir=r'data\4k_12k\4k_12k_with_anno.csv'
    train_data,dev_data,test_data=get_train_and_test_data(anno_dir)
    os.makedirs(DATASET_BASE_DIR,exist_ok=True)
    # pickle_dump(format_filename(DATASET_BASE_DIR,TRAIN_DATA_FILENAME), train_data)
    # pickle_dump(format_filename(DATASET_BASE_DIR,TEST_DATA_FILENAME), test_data)
    # pickle_dump(format_filename(DATASET_BASE_DIR,DEV_DATA_FILENAME), dev_data)


    # 根据训练集和知识库文本，训练词向量，分为char级别和word级别两种
    # prepare character embedding
    # char_vocab：char2idx
    char_vocab, idx2char, char_corpus = load_char_vocab_and_corpus(idx2entity, train_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='char'), char_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='char'), idx2char)
    # c2v为shape=(len(vocabulary) + 2, embedding_dim=300的一个np矩阵
    c2v = train_w2v(char_corpus, char_vocab)
    # c_fastext = train_fasttext(char_corpus, char_vocab) # 目前只支持一种词向量训练方式
    # c_glove = train_glove(char_corpus, char_vocab)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='c2v'), c2v)
    # np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='c_fasttext'), c_fastext)
    # np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='c_glove'), c_glove)

    # 训练词向量需要分词，kb_data里面的alias和subject都作为mention，加入自定义词典
    for mention in mention2entity.keys():
        jieba.add_word(mention, freq=1000000)
    # prepare word embedding
    word_vocab, idx2word, word_corpus = load_word_vocab_and_corpus(idx2entity, train_data)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, VOCABULARY_TEMPLATE, level='word'), word_vocab)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, IDX2TOKEN_TEMPLATE, level='word'), idx2word)
    w2v = train_w2v(word_corpus, word_vocab)
    # w_fastext = train_fasttext(word_corpus, word_vocab)
    # w_glove = train_glove(word_corpus, word_vocab)
    np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='w2v'), w2v)
    # np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='w_fasttext'), w_fastext)
    # np.save(format_filename(PROCESSED_DATA_DIR, EMBEDDING_MATRIX_TEMPLATE, type='w_glove'), w_glove)
