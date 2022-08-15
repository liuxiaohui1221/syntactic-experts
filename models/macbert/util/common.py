import numpy

from models.mypycorrector.utils.text_utils import is_chinese


def stopDuplicateCheck(w,word):
    stopchecks=['队','军','省','市','县','村','每','榜','年','图片','妈妈','由','丝']
    stopwords=['不着急','丝丝']
    if w in stopchecks or word in stopwords:
        return True
    return False

def removeDuplicate(fenci, text):
    arr=fenci.lcut(text)
    # 相邻存在包含关系的
    pre=None
    pre_index=-1
    fine_text = numpy.array(arr)
    for index,word in enumerate(arr):
        fine_text[index] = word
        flag=False
        for w in word:
            if is_chinese(w)==False and stopDuplicateCheck(w,word):
                flag=True
                break
        if flag==False:
            if pre and (len(pre)>1 or len(word)>1):
                if pre and len(word)>len(pre) and word[:len(pre)]==pre:
                    # del pre
                    print("del word:", fine_text[pre_index],"from:",text)
                    fine_text[pre_index]=''
                elif pre and len(pre)>=len(word) and pre[len(pre)-len(word):]==word:
                    # del cur word
                    print("del word:",fine_text[index],"from:",text)
                    fine_text[index]=''
        pre=word
        pre_index=index
    return "".join(fine_text)
