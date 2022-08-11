import jieba
import numpy

from models.model_MiduCTC.src.baseline.ctc_vocab.config import VocabConf


def removeDuplicate(fenci, text):
    arr=fenci.lcut(text)
    # 相邻存在包含关系的
    pre=None
    pre_index=-1
    fine_text = numpy.array(arr)
    for index,word in enumerate(arr):
        fine_text[index] = word
        if pre and (len(pre)>1 or len(word)>1):
            if pre and len(word)>len(pre) and word[:len(pre)]==pre:
                # del pre
                fine_text[pre_index]=''
            elif pre and len(pre)>=len(word) and pre[len(pre)-len(word):]==word:
                # del cur word
                fine_text[index]=''
        pre=word
        pre_index=index
    return "".join(fine_text)

fenci=jieba
text='年年都有余啊'
finetext=removeDuplicate(fenci,text)
print(finetext)