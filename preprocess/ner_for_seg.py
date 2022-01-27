import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG,
                    filename='../log/ner.log',
                    format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
                    # -8表示占位符，让输出左对齐，输出长度都为8位
                    datefmt="%Y-%m-%d %H:%M:%S",
                    encoding='utf8')
from tqdm import tqdm
import stanzaNER
import time


# 填充指定索引区间内的实体
def ner_partation(df,start,end, entity_list, col='mentions'):
    assert len(entity_list)==end-start+1
    df.loc[start:end,col]=entity_list
    return



if __name__ == '__main__':
    df=pd.read_csv(r'data/segment_articles.csv',sep='\t',index_col=0) # 省略的当前目录即pwd
    # max_line=df.shape[0]
    max_line=2050 # 先测试一下2050行/100,耗时262秒
    t_per_epoch=500
    epochs = max_line // t_per_epoch + 1 # 101轮循环完
    logging.info('NER START:总行数:{},每轮迭代数量:{},总轮次:{}'.format(max_line,t_per_epoch,epochs))
    start_t=time.time()
    for i in tqdm(range(epochs)):
        try:
            logging.debug('当前进度=====epoch:{}/{}====='.format(i+1,epochs))
            start = i*t_per_epoch
            end = min((i+1)*t_per_epoch,max_line)-1
            texts=df.loc[start:end, 'text'].tolist()
            entitys=stanzaNER.ner(texts)
            ner_partation(df,start,end, entitys, col='mentions')
            if i%10==0:
                logging.info('epoch {},临时存储data'.format(i+1))
                df.to_csv('segment_articles_tmp_end_{}.csv'.format(end),sep='\t')
        except Exception:
            logging.error('发生异常，当前start:{}, 耗时:{:.2f}秒'.format(start,time.time() - start_t))
            df.to_csv('segment_articles_test0122_start{}.csv'.format(start),sep='\t')
    logging.info("所有文档实体识别完成，总耗时: {:.2f}秒".format(time.time() - start_t))
    df.to_csv('segment_articles_test0122.csv',sep='\t')

