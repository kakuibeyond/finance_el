import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, filename='ner_log')
from tqdm import tqdm
import stanzaNER
import time



def ner_partation(df,start,end, entity_list, col='mentions'):
    assert len(entity_list)==end-start+1
    df.loc[start:end,col]=entity_list
    return



if __name__ == '__main__':
    df=pd.read_csv(r'..\data\segment_articles.csv',sep='\t',index_col=0)
    # max_line=df.shape[0]
    max_line=2050 # 先测试一下2050行,耗时300.87秒
    t_per_epoch=1000
    epochs = max_line // t_per_epoch + 1 # 101轮循环完
    start_t=time.time()
    for i in tqdm(range(epochs)):
        try:
            start = i*t_per_epoch
            end = min((i+1)*t_per_epoch,max_line)-1
            texts=df.loc[start:end, 'text'].tolist()
            entitys=stanzaNER.ner(texts)
            ner_partation(df,start,end, entitys, col='mentions')
        except Exception:
            logging.error('发生异常，当前start:{}, 耗时:{:.2f}秒'.format(start,time.time() - start_t))
    print("执行成功，耗时: {:.2f}秒".format(time.time() - start_t))

