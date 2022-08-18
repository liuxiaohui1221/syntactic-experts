import os

import models.model_MiduCTC.src.thulac as thulac
# thu1 = thulac.thulac(seg_only=True)  #只进行分词，不进行词性标注
from ProjectPath import get_project_path

# thu1 = thulac.thulac(seg_only=True)
from models.model_MiduCTC.src.baseline.ctc_vocab.config import VocabConf

thu1=VocabConf().thulac_singleton
#
texts1=thu1.cut("还代款")
# texts2=thu1.cut("被判处刑罚的，还要开除公职。")
predict_text=thu1.cut('还贷款')

print(texts1)
# print(texts2)
print(predict_text)

ws="澳门科技大学大一学生钱淳正说，如今我们拥有了自己的空间站，相信未来，中国人的脚步一定会踏入更远深空。"
ws="延申阅读记者：张虞李鹏志编辑：胡梦莹编审：赵宗杰终审：李恩广总监制：肇慧茹"
print(len(ws))
print(thu1.cut(ws))


