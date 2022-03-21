# -*- coding: utf-8 -*-

from os import path
from torch.optim import Adam
import torch

# RAW_DATA_DIR = './raw_data'
PROCESSED_DATA_DIR = './data/deal_pkl'
LOG_DIR = './log'
MODEL_SAVED_DIR = './ckpt'

# CCKS_DIR = path.join(RAW_DATA_DIR, 'ccks2019_el')
# CCKS_TRAIN_FILENAME = path.join(CCKS_DIR, 'train.json')
# CCKS_TEST_FILENAME = path.join(CCKS_DIR, 'develop.json')
# CCKS_TEST_FINAL_FILENAME = path.join(CCKS_DIR, 'eval722.json')
# KB_FILENAME = path.join(CCKS_DIR, 'kb_data')

# 训练测试数据存储
# TRAIN_FILE_SET=['train','test','dev']
DATASET_BASE_DIR='data/dataset'
TRAIN_DATA_FILENAME = 'erl_train.pkl'
DEV_DATA_FILENAME = 'erl_dev.pkl'
TEST_DATA_FILENAME = 'erl_test.pkl'
DATASET_FILENAME={'train':TRAIN_DATA_FILENAME,
                    'dev':DEV_DATA_FILENAME,
                    'test':TEST_DATA_FILENAME}
# TEST_FINAL_DATA_FILENAME = 'erl_test_final.pkl'

# 实体相关信息存储位置
ENTITY_DESC_FILENAME = 'entity_desc.pkl'# 暂未使用
MENTION_TO_ENTITY_FILENAME = 'mention_to_entity.pkl'
ENTITY_TO_MENTION_FILENAME = 'entity_to_mention.pkl' # 暂未使用
ENTITY_TO_IDX = 'entity_to_idx.pkl'
IDX_TO_ENTITY = 'idx_to_entity.pkl'
# ENTITY_TYPE_FILENAME = 'entity_type.pkl'


VOCABULARY_TEMPLATE = '{level}_vocab.pkl'
IDX2TOKEN_TEMPLATE = 'idx2{level}.pkl'
EMBEDDING_MATRIX_TEMPLATE = '{type}_embeddings.npy'
PERFORMANCE_LOG = '{model_type}_performance.log'


PRETRAIN_MODEL_NAME={'bert':'bert-base-chinese',
            'wwm':'hfl/chinese-bert-wwm',
            'ernie':'nghuyong/ernie-1.0'}

# EXTERNAL_EMBEDDINGS_DIR = path.join(RAW_DATA_DIR, 'embeddings')
# EXTERNAL_EMBEDDINGS_FILENAME = {
#     'bert': path.join(EXTERNAL_EMBEDDINGS_DIR, 'chinese_L-12_H-768_A-12'),
#     'ernie': path.join(EXTERNAL_EMBEDDINGS_DIR, 'baidu_ernie'),
#     'bert_wwm': path.join(EXTERNAL_EMBEDDINGS_DIR, 'chinese_wwm_L-12_H-768_A-12')
# }

MAX_SEQ_LEN=300
MAX_DESC_LEN=50

class ModelCofig(object):
    def __init__(self) -> None:
        # input base config
        self.embed_dim = 300 # 百度词向量维度
        self.embed_trainable = True
        self.embeddings = None
        self.vocab = None   # character embedding as base input
        self.vocab_size = None
        self.mention_to_entity = None
        self.entity_desc = None

        self.use_char_input = True

        self.use_bert_input = False
        self.bert_type = 'bert'
        # self.bert_model_file = lambda x: path.join(EXTERNAL_EMBEDDINGS_FILENAME[x], 'bert_model.ckpt')
        # self.bert_config_file = lambda x: path.join(EXTERNAL_EMBEDDINGS_FILENAME[x], 'bert_config.json')
        # self.bert_vocab_file = lambda x: path.join(EXTERNAL_EMBEDDINGS_FILENAME[x], 'vocab.txt')
        # self.bert_layer_num = 1
        # self.bert_seq_len = 50
        self.bert_trainable = True

        # self.use_bichar_input = False
        # self.bichar_vocab = None
        # self.bichar_embed_dim = 50
        # self.bichar_embed_trainable = False
        # self.bichar_embeddings = None
        # self.bichar_vocab_size = None

        # self.use_word_input = False
        # self.word_vocab = None
        # self.word_embed_dim = 300
        # self.word_embed_trainable = False
        # self.word_embeddings = None
        # self.word_vocab_size = None

        # self.use_charpos_input = False
        # self.charpos_vocab = None
        # self.charpos_embed_dim = 300
        # self.charpos_embed_trainable = False
        # self.charpos_embeddings = None
        # self.charpos_vocab_size = None

        # self.use_softword_input = False
        # self.softword_embed_dim = 50

        # self.use_dictfeat_input = False

        # self.use_maxmatch_input = False
        # self.maxmatch_embed_dim = 50

        # input config for entity linking model
        self.max_cand_mention = 10
        self.max_cand_entity = 10
        self.max_desc_len = 400
        self.use_relative_pos = False  # use relative position (to mention) as input
        self.max_erl_len = 30
        self.n_rel_pos_embed = 60
        self.rel_pos_embed_dim = 50
        self.omit_one_cand = True
        self.n_neg = 1

        # model structure configuration
        self.exp_name = None # 由所有参数拼接起来的模型名称
        self.model_name = None # 模型自定义名称
        self.rnn_units = 300
        self.dense_units = 128

        # model training configuration
        self.batch_size = 64
        self.n_epoch = 50
        self.learning_rate = 0.001
        self.optimizer = Adam(self.learning_rate)
        self.threshold = 0.5
        self.logging_steps = 10 # 每10个step记录一下
        self.save_steps = 10
        self.f1_threshold = 0.6

        # checkpoint configuration
        self.checkpoint_dir = MODEL_SAVED_DIR

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        