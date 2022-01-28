# -*- coding: utf-8 -*-
import stanza
import jieba
import time
import logging
logging.basicConfig(level=logging.INFO)
download_dir='D:\data\stanza_resources'
# stanza.download('zh',model_dir=download_dir)


# 可以通过pipeline预加载不同语言的模型，也可以通过pipeline选择不同的处理模块，还可以选择是否使用GPU：
zh_nlp = stanza.Pipeline('zh', processors='tokenize,ner,pos', 
                                tokenize_pretokenized=True,
                                use_gpu=False,
                                dir=download_dir)


# def ner(text):
#     print('len of text:',len(text))
#     doc = zh_nlp(text)
#     for i, sentence in enumerate(doc.sentences):
#         print(f'====== Sentence {i+1} tokens =======')
#         print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')

def ner_single(single_text):
    logging.info('len of current doc:',len(single_text))
    text_w=[jieba.lcut(single_text) ]
    res=[]
    keep_type=['GPE','LOC','PERSON','ORG']
    logging.info('开始加载文档===')
    doc = zh_nlp(text_w)
    logging.info('===结束加载文档')
    assert len(doc.sentences)==1
    sent=doc.sentences[0]
    curr_en=set()# 当前句子实体集合
    for ent in sent.ents:
        if ent.type in keep_type:
            t=ent.text.replace(' ','')
            curr_en.add(f'{t}/{ent.type}')
    if len(curr_en)>0:
        return ';'.join(curr_en)
    else:
        return 'O'

# 文档列表，返回与文档列表等长的标签列表，句子无实体用'O'表示
def ner(docs):
    logging.info('=====len of doc:{}====='.format(len(docs)))
    t1=time.time()
    text_w=[jieba.lcut(t) for t in docs]
    t2=time.time()
    logging.debug('分词耗时:{:.2f}秒'.format(t2-t1))
    res=[]
    keep_type=['GPE','LOC','PERSON','ORG'] #PER（人物），LOC（地点），ORG（组织），GPE（地缘政治实体（geo-political entity））
    # logging.info('===开始加载文档===')
    doc = zh_nlp(text_w)
    t3=time.time()
    logging.debug('加载文档耗时:{:.2f}秒'.format(t3-t2))
    # logging.info('===结束加载文档===')
    for i, sent in enumerate(doc.sentences):
        # print("Sentence: " + sent.text)  # 因为提前分词，所以这里文本（自带空格分割）和后面分词结果打印出来一模一样
        # print("Tokenize：" + '||'.join(token.text for token in sent.tokens))  # 中文分词
        curr_en=set()# 当前句子实体集合 无重复
        for ent in sent.ents:
            if ent.type in keep_type:
                t=ent.text.replace(' ','')
                curr_en.add(f'{t}/{ent.type}')
        if len(curr_en)>0:
            res.append(';'.join(curr_en))
        else:
            res.append('O')
    assert len(res)==len(docs)
    logging.debug('筛选实体耗时:{:.2f}秒'.format(time.time()-t3))
    return res
    
# res=ner(text)
# print('len of final entities size:',len(res))
# print(res[-3:])
# print(text[618:622])
# print(text[3243:])

if __name__ == '__main__':
    text = """推动消费金融行业井喷式增长
    “现在工作这么累，想在休息日好好犒劳自己。我会去世界各地旅行，已经去过韩国、日本和好几个欧洲国家，土耳其、以色列、厄瓜多尔的加拉帕戈斯群岛都在我未来的旅行清单上。”上海某律所的年轻律师吴辰伟对记者说，尽管自己收入还不错，但动辄两三万元的旅行花费也让他觉得有些头疼。所以今年“十一”假期和女友去塞班岛度假，吴辰伟在去哪儿旅行网预订时就选择了分期付款。“包含往返机票和五星级酒店住宿的7天塞班岛自由行，如果一次性付款，两个人总价是22100多元，我选择的是分12期付款，每期还1960元左右，这种付款方式为我减轻了用钱压力。不一定赚多少才能花多少，‘先消费后付款’很受我们年轻人欢迎。”吴辰伟说,
    80后、90后等年轻一代消费方式的改变和信用消费意识的增强，是近些年我国消费金融行业蓬勃发展的重要动力。来自山东临沂的刘明露今年24岁，在天津一家创业公司担任行政助理。“我平时喜欢看日本动漫和综艺，慢慢地对日语产生了兴趣。但是我收入不高，扣除房租和日常开销后，工资所剩有限，没办法一次性付清学日语的钱。”刘明露说，近两年她喜欢上了“先消费后付款”的模式。去年她用蚂蚁花呗在淘宝上分24期购买了一部华为P10手机，今年3月用招行信用卡刷了一台联想笔记本电脑。“这次我报名沪江网校的日语课用的是京东白条，分6期免息，每月还款510元。以后我还想学学商务英语，只要在自己的承受范围内且能及时还款，花未来的钱满足今天的愿望，我觉得是一笔划算的买卖。”
    何飞说，形成金融创新促进消费升级的良性循环，关键还是要坚持“开正门”“堵偏门”。既要鼓励各正规消费金融机构加大力度，以满足更丰富的消费场景为出发点推进金融产品创新，也要始终按照金融业务必须持牌经营的原则，推动不具备从业资质的平台加快市场出清，坚决遏制打着互联网创新旗号、披着技术外衣违法违规开展金融业务的行为，实现正本清源和良性发展。
    (责编：杜燕飞、董菁)"""
    text=list(text.split('\n'))
    # jieba.add_word('金融行业', freq=1000000)
    
    res=ner(text)
    print(res)
    
    # test single
    # text = '土耳其、以色列、厄瓜多尔的加拉帕戈斯群岛都在我未来的旅行清单上。”上海某律所的年轻律师吴辰伟对记者说'
    # print(ner_single(text))