import os

import jieba

from ProjectPath import get_project_path

user_dict = "knowledgebase/dict/custom_dict.txt"
user_dict=os.path.join(get_project_path(),user_dict)
# jieba.load_userdict(user_dict)

ws="同袍共泽"
seg_list = jieba.cut(ws, cut_all=False)  # 使用精确模式进行分词
print("/".join(seg_list))