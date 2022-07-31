
import json
import os.path
import random

import numpy as np
from tqdm import tqdm

from ProjectPath import get_project_path
from knowledgebase.char_sim import CharFuncs
from knowledgebase.chinese_pinyin_util import ChinesePinyinUtil
from knowledgebase.chinese_shape_util import ChineseShapeUtil


def isChinese(cchar):
    if u'\u4e00' <= cchar <= u'\u9fff':
        return True
    else:
        return False
class DataGegerator:
    def __init__(self,seed):
        self.data_out_path = os.path.join(get_project_path(), "model/model_MiduCTC/data/preliminary_a_data")
        self.train_path = os.path.join(self.data_out_path, 'preliminary_extend_train.json')
        self.train_data = json.load(open(self.train_path, encoding='utf-8'))
        self.pinyin_util= ChinesePinyinUtil()
        self.chinese_shape_util = ChineseShapeUtil()
        random.seed(seed)

    # 80%一个文本只出现一个错误，20%出现5个以内
    # 70%拼音相似，20%形状相似，10%两者结合等

    def chooseChineseWithPinyin(self,choosedChinese,topN=1):
        chineses = self.pinyin_util.getSimilarityChineseByPinyin(choosedChinese)
        if len(chineses)==0:
            return None
        # 80%:同音且形近中选
        p=random.randint(0,100)
        if p<80:
            # 根据同拼音对应汉字列表，分别求与原汉字形似度，最相似列表中top3随机
            resultChineses = self.chinese_shape_util.getTopSimilarityShapeFromBackup(choosedChinese,chineses)
            finalChinese = resultChineses[random.randint(0, min(topN, len(resultChineses) - 1))]
            return finalChinese
        return chineses[random.randint(0,len(chineses)-1)]

    def chooseChineseWithShape(self,choosedChinese,topN=2):
        simShapeChineses=self.chinese_shape_util.getAllSimilarityShape(choosedChinese,topN=3)
        if len(simShapeChineses)==0:
            return None
        # topN中随机
        return simShapeChineses[random.randint(0,min(topN,len(simShapeChineses)-1))]

    def chooseChineseWithPinyinAndShape(self,choosedChinese):
        # 音近:从音近列表随机选一个拼音,根据拼音获取常用汉字列表
        simChineses=self.pinyin_util.getSimilarityChineseBySimPinyin(choosedChinese)
        if len(simChineses)==0:
            return None
        # 随机常用汉字
        return simChineses[random.randint(0,len(simChineses)-1)]

    def getNewText(self,text, pos, simChinese):
        if pos==0:
            return simChinese+text[pos+1:]
        elif pos==len(text)-1:
            return text[:-1]+simChinese
        else:
            return text[:pos]+simChinese+text[pos+1:]

    def data_generator(self,simPinyinPercent=50,equalPinyinPercent=40,shapePercent=10,generateOnePercent=90,generateMaxForOne=5):
        text_gens=[]
        for ins in tqdm(self.train_data[:]):
            text=ins['target']
            iter=1
            chooseChoice=random.randint(0,100)
            if chooseChoice>=generateOnePercent:
                iter=random.randint(1,generateMaxForOne)
            alreadyCheckedPos=np.zeros(len(text),dtype=int)
            for i in range(iter):
                choosedChinese=None
                pos=-1
                for i in range(5):
                    pos=random.randint(0,len(text)-1)
                    if self.isChinese(text[pos]) and alreadyCheckedPos[pos]==0:
                        choosedChinese=text[pos]
                        alreadyCheckedPos[pos]=1
                        break
                if choosedChinese==None:
                    continue

                p=random.randint(0,simPinyinPercent+equalPinyinPercent+shapePercent)
                if p<equalPinyinPercent:
                    # choose similarity with pinyin
                    simChinese=self.chooseChineseWithPinyin(choosedChinese)
                elif p>=equalPinyinPercent and p<equalPinyinPercent+shapePercent:
                    simChinese=self.chooseChineseWithShape(choosedChinese)
                else:
                    simChinese=self.chooseChineseWithPinyinAndShape(choosedChinese)
                if simChinese == None:
                    continue
                text_gen=self.getNewText(text,pos,simChinese)
                text_gens.append({
                    "id": ins['id'] + 10000000,
                    "source": text_gen,
                    "target": ins['target'],
                    "type": "negative"
                })
        return text_gens
if __name__ == '__main__':
    dg=DataGegerator(121)
    outFile = 'preliminary_extend_train_gen3.json'
    # 50%近似音（包括同音）中随机，40%同音字中随机，10%形近字top3中随机
    text_gens=dg.data_generator(generateOnePercent=1,simPinyinPercent=15,equalPinyinPercent=80,shapePercent=5)
    json.dump(text_gens, open(os.path.join(dg.data_out_path, outFile), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)