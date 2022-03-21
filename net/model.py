import imp
import torch
import torch.nn as nn
from config import ModelCofig
from module import Features_Link
from transformers import BertModel
# from .task import Locate_Entity, Link_KB
# from .dataset import text2bert, seqs2batch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score

class Link_KB(nn.Module):
    """
    输入 实体特征、知识特征，预测两者链接得分
    """

    def __init__(self,
                 hidden_dim):
        super(Link_KB, self).__init__()

        # 实体和知识库拼接，只做二分类
        self.cal_score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.hidden_dim = hidden_dim


    def forward(self,
                entity_features,
                kb_features,
                labels=None):
        entity_features_ = nn.MaxPool1d(entity_features.size()[1])(entity_features.transpose(2, 1)).squeeze(-1)
        kb_features_ = nn.MaxPool1d(kb_features.size()[1])(kb_features.transpose(2, 1)).squeeze(-1)

        features = entity_features_ * kb_features_
        features = torch.cat([entity_features_, kb_features_, features], -1)

        scores = self.cal_score(features).squeeze(-1)

        loss=None
        if labels:
            # 计算实体和知识库拼接得分的损失
            loss = nn.BCEWithLogitsLoss()(scores, labels)

        return scores,loss


# 实体链接模型
class LinkingModel(nn.Module):
    def __init__(self, config: ModelCofig):
        super(LinkingModel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_type)
        if not config.bert_trainable:
            for p in self.bert.embeddings.parameters():
                p.requires_grad = False

        # 实体链接得分
        self.link_kb = Link_KB(hidden_dim=config.dense_units)



    def cal_link_loss(self,
                    text_ids=None,
                    desc_ids=None,
                    labels=None
                    ):
        text_features = self.bert(**text_ids) # text_ids是一个字典 内部已经封装了bert需要的输入

        desc_features = self.bert(**desc_ids)

        # 计算链接得分的损失
        _, loss = self.link_kb(text_features,
                            desc_features,
                            labels)

        return loss


    # 验证集计算得分和loss，这里先用文本匹配的思路做，后期改为从所有cand中寻找最高得分的entity_id
    def evaluate(self,devloader,check_point=''):
        if check_point != '':
            pth=torch.load(check_point,map_location='cpu')
            self.load_state_dict(pth['weights'])
        result ={}
        eval_loss = 0.0
        nb_eval_steps = 0
        y_pred=[] # 预测的匹配标签 0、1
        y_true=[]
        self.eval()
        with torch.no_grad():
            for val_batch in tqdm(devloader, desc="Evaluating"):
                nb_eval_steps += 1
                val_batch={k:v.to(self.config.device) for k,v in val_batch.items()}
                logits,loss = self.forward(**val_batch)
                y_pred.extend(torch.argmax(logits,dim=-1).detach().cpu().tolist())
                y_true.extend(val_batch['labels'].cpu().tolist())
                eval_loss + loss.mean().item()

            p = precision_score(y_true, y_pred, average='binary')
            r = recall_score(y_true, y_pred, average='binary')
            f1 = f1_score(y_true, y_pred, average='binary')
            acc = accuracy_score(y_true,y_pred)

            eval_loss = eval_loss / nb_eval_steps
        result = {'p':p,'r':r,'f1':f1,'acc':acc,'evl_ls':eval_loss}
        return result

    def forward(self,
                text_ids=None,
                desc_ids=None,
                labels=None):
        
        loss=None
        if labels is not None:
            pass
            # loss=

        # 链接知识库
        results = []
        for entity_info in entity_predict:
            entity, s, e = entity_info

            # 计算做链接的语义
            entity_features = self.get_features_link(text_seq)

            if entity in alias_data:
                alias_predict = alias_data[entity.lower()]
                kb_ids = list(alias_predict.keys())

                num_batch = (len(kb_ids) - 1) // 64 + 1
                link_scores = []
                for n in range(num_batch):
                    kb_ids_batch = kb_ids[(n * 64):(n * 64 + 64)]
                    kb_seqs = [alias_predict[kb_id]['data_seq'] for kb_id in kb_ids_batch]
                    kb_seqs, kb_max = seqs2batch(kb_seqs)
                    kb_seqs = torch.LongTensor(kb_seqs).to(self.device)

                    # 计算kb文本语义
                    kb_features = self.get_features_link(kb_seqs)

                    # 计算batch链接得分
                    text_features_batch = entity_features.repeat(len(kb_ids_batch), 1, 1)
                    link_scores_batch = nn.Sigmoid()(self.link_kb(text_features_batch, kb_features)).tolist()
                    link_scores += link_scores_batch
                score = max(link_scores)
                results.append((kb_ids[link_scores.index(score)], entity, s, e, score))

        return results
