# finance_el

## 1.预处理操作

- 文档分段，得到`data\segment_articles.csv`
- 分词,`NER`，得到`data\segment_articles_0127_with_entity.csv`
- 候选实体生成，mention2entity映射字典为`data\m2e.pkl`
- 生语料用于标注，`data\seg_anno.csv`
