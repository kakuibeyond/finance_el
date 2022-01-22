# -*- coding: utf-8 -*-

# from transformers import pipeline

# ner_pipe = pipeline("ner")
# sequence = """Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO,
# ... therefore very close to the Manhattan Bridge which is visible from the window."""
# for entity in ner_pipe(sequence):
#     print(entity)

# d={}
# for i,s in enumerate(items):
#     d[i]=s
#     print(i,':',s,',')
# print(d)


docs=[]
with open('test_doc.txt','r',encoding='utf-8') as f:
    docs = f.readlines()
# print(docs[-1])
# print(docs[0])
from aip import AipNlp
import json

""" 你的 APPID AK SK """
APP_ID = '25501021'
API_KEY = 'gGGUZIDklNLuDfeslZlWcmRS'
SECRET_KEY = 'qxq6WpMX2wO23MgrELcXvBieZ3GDaFmj'

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
text = "百度和快手都是总部位于北京的互联网企业。"
charset="UTF-8"

""" 调用词法分析 """
data = client.lexer(text)
print(data)
entitys=[]
if "error_code" not in data or data["error_code"] == 0:
    for item in data["items"]:
        if len(item['ne'])>0:
            curr={"ne":item['ne'],"uri":"","mention":item['item'],"offset":item['byte_offset']}
            entitys.append(curr)
print(entitys)