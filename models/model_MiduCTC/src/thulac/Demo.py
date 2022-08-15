import os

import models.model_MiduCTC.src.thulac as thulac
# thu1 = thulac.thulac(seg_only=True)  #只进行分词，不进行词性标注
from ProjectPath import get_project_path

thu1 = thulac.thulac(seg_only=True)
#
texts1=thu1.cut("还代款")
# texts2=thu1.cut("被判处刑罚的，还要开除公职。")
predict_text=thu1.cut('还贷款')

print(texts1)
# print(texts2)
print(predict_text)

ws="捐款有限、爱心无限，大家通过现场捐款方式以实际行动支援河南灾区工作，切实发挥党员先锋模范作用，充分弘扬“一方有难，八方支援”的传统美德，体现了检察人员与灾区人民“情相牵、心相连”共度难关的坚定决心。"
print(len(ws))
print(thu1.cut(ws,text=True))


