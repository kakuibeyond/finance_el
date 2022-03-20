# finance_el

## 1.预处理操作

- 文档分段，得到`data\segment_articles.csv`
- 分词,`NER`，得到`data\segment_articles_0127_with_entity.csv`
- 候选实体生成，mention2entity映射字典为`data\m2e.pkl`
- 生语料用于标注，`data\seg_anno.csv`，读取方式

```python
df_r=pd.read_csv(anno_dir,sep='\t',index_col=0)
```

- 临时标注后的结果存储位置(如`data\4k_12k\`)，读取方式同上

## 语料读取

**读取方式**

```python
before_anno_dir='..\data\seg_anno.csv'
df=pd.read_csv(before_anno_dir,sep='\t',index_col=0)
df.head()
```

- index：索引列
- srcc_idx：所属原始文档（篇章）
- keyword：文档关键词
- text：文本内容
- mention：实体提及
- entity_list：候选实体列表
- annotation：标注正确的实体指向
