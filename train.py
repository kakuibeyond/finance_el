import pickle
import os
import torch
import torch.optim as optim
import torch.utils.data.DataLoader as DataLoader
from sklearn.model_selection import train_test_split
from net.LinkingDataset import LinkingDataset,collate_fn_link
from net.model import LinkingModel
import pandas as pd
from transformers import BertTokenizer, BertModel,get_scheduler
# import argparse
from utils.io import pickle_load,format_filename
from config import *
from tqdm.auto import tqdm
import json
from time import time
import logging
logger = logging.getLogger(__name__)

from torch.utils.tensorboard import SummaryWriter


# parser = argparse.ArgumentParser()

# parser.add_argument('--cuda', default='0', help='cuda:0/1/2')
# parser.add_argument('--pretrain', default='bert', help='bert,wwm,ernie')
# parser.add_argument('--num_layers', default=4, type=int, help='lstm layernum 3/4')
# parser.add_argument('--hidden_dim', default=768, type=int, help='lstm hidden 768/1024')
# parser.add_argument('--loss_weight', default=3, type=int, help='loss:2/3')
# parser.add_argument('--num_words', default=10000, type=int, help='num_words:9000/10000')
# parser.add_argument('--max_len', default=400, type=int, help='max_len:300/400/500')
# parser.add_argument('--epochs', default=10, type=int, help='epochs')
# parser.add_argument('--ner_id', default=1, type=int, help='None')
# parser.add_argument('--k', default=0.813, type=float, help='k')
# parser.add_argument('--lr', default=0.001, type=float, help='k')
# parser.add_argument('--n', default=1, type=int, help='n')

# opt = parser.parse_args()


# torch.manual_seed(1)



def train_link(model_name, batch_size=32, n_epoch=50, learning_rate=0.001, optimizer_type='adam',
                bert_type=None, bert_trainable=True, n_neg=1, omit_one_cand=True):
    config = ModelCofig()
    config.model_name = model_name
    config.batch_size = batch_size
    config.n_epoch = n_epoch
    config.learning_rate = learning_rate
    config.bert_type = bert_type
    config.bert_trainable = bert_trainable
    config.mention_to_entity = pickle_load(format_filename(PROCESSED_DATA_DIR, MENTION_TO_ENTITY_FILENAME))
    config.entity_desc = pickle_load(format_filename(PROCESSED_DATA_DIR, IDX_TO_ENTITY)) #使用实体名称代替desc

    config.exp_name = '{}_{}_{}_{}_{}_{}'.format(model_name, bert_type,'tune' if bert_trainable else 'fix',
                                                 batch_size, optimizer_type,
                                                 learning_rate)
    config.n_neg = n_neg
    if config.n_neg > 1:
        config.exp_name += '_neg_{}'.format(config.n_neg)
    config.omit_one_cand = omit_one_cand
    if not config.omit_one_cand:
        config.exp_name += '_not_omit'
    
    model_save_path = os.path.join(config.checkpoint_dir, config.exp_name)
    os.makedirs(model_save_path,exist_ok=True)

    train_log = {'exp_name': config.exp_name, 'batch_size': batch_size, 'optimizer': optimizer_type, 'epoch': n_epoch,
                 'learning_rate': learning_rate}

    print('Logging Info - Experiment: %s' % config.exp_name)
    model = LinkingModel(config).to(config.device)
    

    tokenizer = BertTokenizer.from_pretrained(PRETRAIN_MODEL_NAME[bert_type])

    trainloader = DataLoader(dataset=LinkingDataset(file_mode='train',entity_desc=config.entity_desc,n_neg=n_neg),
                             batch_size=config.batch_size,
                             shuffle=True,
                             collate_fn=lambda x: collate_fn_link(x, tokenizer))
    devloader = DataLoader(dataset=LinkingDataset(file_mode='dev',entity_desc=config.entity_desc,n_neg=n_neg),
                            batch_size=config.batch_size,
                            shuffle=False,
                            collate_fn=lambda x: collate_fn_link(x, tokenizer))

    num_training_steps = n_epoch * len(trainloader)
    progress_bar = tqdm(range(num_training_steps))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=learning_rate)
    lr_scheduler = get_scheduler("linear",
                                optimizer=optimizer,
                                num_warmup_steps=0,
                                num_training_steps=num_training_steps)

    # 保存最优的3个模型
    f1_scores=[0,0,0]
    model.train()
    global_step=0
    tb_writer = SummaryWriter()
    start_time=time()
    for epoch in n_epoch:
        total_loss = 0
        print("epoch:{},curr_step:{},total_step:{}".format(epoch,global_step,num_training_steps))
        for batch in trainloader:
            global_step+=1
            batch={k:v.to(config.device) for k,v in batch.items()}
            loss, outputs = model(**batch)
            total_loss += loss.item()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            #每logging_steps，进行evaluate
            if global_step % config.logging_steps:
                logs={}
                results = model.evaluate(devloader)
                for key, value in results.items():
                    eval_key = 'eval_{}'.format(key)
                    logs[eval_key] = value
                for key, value in logs.items():
                    tb_writer.add_scalar(key, value, global_step)
                print(json.dumps({**logs, **{'step': global_step}}))
                if results['f1']>max(f1_scores[-1],config.f1_threshold):
                    f1_scores.append(results['f1'])
                    torch.save({
                                'epoch':epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict':optimizer.state_dict(),
                                'loss':total_loss/(global_step*batch_size)
                                },os.path.join(model_save_path,'gstep_{}f1_{:.4f}.pt'.format(global_step,f1_scores[-1])))
                    f1_scores=f1_scores[1:]
            #每save_steps保存checkpoint
            # if global_step % config.save_steps:
    tb_writer.close()

if __name__ == '__main__':
    train_link()
    