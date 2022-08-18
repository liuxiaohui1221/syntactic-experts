import os

import jieba

from ProjectPath import get_project_path
from models.model_MiduCTC.src.baseline.ctc_vocab.config import VocabConf

ws=[
    "就是以联合国宪章宗旨和原则为基础的国际关系基本准则",
    "还代款",
    "立足法律监督智能",
    "集中处治",
    "以咬定青山不放松的执着、行百里者半九十的清醒不懈奋斗，敢于斗争、善于斗争，逢山开道、遇水架桥，中国人民孜孜以求的美好梦想终将成为现实。",
    "江苏省长吴政隆日前到扬州检查指导疫情防控工作，他说这次扬州疫情发生早、发现比较晚，在人员聚集的密闭场所，老年人居多，现在情况尚未见底。"
]
# user_dict = "knowledgebase/dict/custom_dict.txt"
# user_dict=os.path.join(get_project_path(),user_dict)
# jieba.load_userdict(user_dict)
my_jieba=VocabConf().jieba_singleton
for text in ws:
    seg_list = my_jieba.lcut(text, cut_all=False)  # 使用精确模式进行分词
    print("/".join(seg_list))

    seg2=my_jieba.lcut(text,cut_all=True)
    print("/".join(seg2))
