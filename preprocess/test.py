# import os,sys
# import pandas as pd
# import baiduNER 
# if __name__ == '__main__':
#     currentUrl = os.path.dirname(__file__)#当前 xxx.py 文件路径
#     print('currentUrl:',currentUrl)
#     # parentUrl = os.path.abspath(os.path.join(currentUrl, os.pardir))#上级路径 注意其中os.pardir也可换成'..'表示
#     # print('parentUrl:',parentUrl)
#     # file_name=r'..\data\test_500.csv'
#     # df=pd.read_csv(file_name,sep='\t')
#     # print(df.shape)
#     print('cwd:',os.getcwd())
#     print('curdir',os.curdir)
#     print('sys_path',sys.path)

import pickle
# m2e_dir=r'data\deal_pkl\mention_to_entity.pkl'
m2e_dir=r'data\m2e_all.pkl'
with open(m2e_dir,'rb') as fr2: # 字典key为mention，value为候选实体列表的字典
    m2e=pickle.load(fr2)
num_men=len(m2e)
entity_set=set()
num_entity=0
nn=0
for k,v in m2e.items():
    n=len(v)
    # if k=='阿里巴巴集团':
    #     print('阿里指向实体数量：{}'.format(len(v)))
    #     for vv in v: print(vv)
    if n==5 and nn<20:
        print('mention:{},entity:{}'.format(k,v))
        nn+=1
    entity_set.update(v)
    num_entity+=len(v)
print('mention数量总计{}个，entity数量总计{}个，平均每个实体指称指向实体数量为{}个'.format(num_men,len(entity_set),num_entity/num_men))