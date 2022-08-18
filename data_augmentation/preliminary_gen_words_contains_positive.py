
import json
import os.path
import random

import jieba
import numpy as np
from tqdm import tqdm

from ProjectPath import get_project_path
from knowledgebase.char_sim import CharFuncs
from knowledgebase.chinese_pinyin_util import ChinesePinyinUtil
from knowledgebase.chinese_shape_util import ChineseShapeUtil
from knowledgebase.dict.gen_confusion import readConfusions, readEasyConfusionWord
from models.model_MiduCTC.src.baseline.ctc_vocab.config import VocabConf
from models.model_MiduCTC.src.thulac import thulac
from models.mypycorrector.proper_corrector import ProperCorrector

'''
1.同音，近音，形近
2.先分词，对词组进行同音或近音替换。若在混淆集中，则优先从混淆集中选择替换
'''

# 过滤128长度后的词
def filterSingleWord(src_words):
    fine_grans=[]
    single_grans=[]
    n=0
    for words in src_words:
        n+=len(words)
        if len(words)==1:
            if isChinese(words):
                single_grans.append(words)
            continue
        else:
            flag=False
            for word in words:
                if isChinese(word)==False:
                    flag=True
                    break
            if flag:
                continue
        if n>128:
            break
        fine_grans.append(words)
    return fine_grans,single_grans
def isChinese(cchar):
    if u'\u4e00' <= cchar <= u'\u9fff':
        return True
    else:
       return False

class DataGegerator:
    def __init__(self,seed,dataFile,basePath="models/ECSpell/Data/traintest"):
        self.data_out_path = os.path.join(get_project_path(), basePath)
        self.train_path = os.path.join(self.data_out_path, dataFile)
        self.train_data = json.load(open(self.train_path, encoding='utf-8'))
        self.pinyin_util= ChinesePinyinUtil()
        self.chinese_shape_util = ChineseShapeUtil()
        self.jieba=VocabConf().jieba_singleton
        proper_path = os.path.join(get_project_path(), 'knowledgebase/dict/custom_dict.txt')
        self.proper=ProperCorrector(proper_name_path=proper_path)
        confusion_path = os.path.join(get_project_path(), 'models/mypycorrector/data/confusion_pair.txt')
        self.confusions=readConfusions(inPath=confusion_path)
        # 加载易错词
        self.easyerr_words=readEasyConfusionWord()
        random.seed(seed)


    def chooseChineseWithPinyinAndShape(self,choosedChinese,topN=3):
        chineses = self.pinyin_util.getSimilarityChineseByPinyin(choosedChinese)
        if len(chineses) == 0:
            return None
        # 根据同拼音对应汉字列表，分别求与原汉字形似度，最相似列表中topk
        resultChineses = self.chinese_shape_util.getTopSimilarityShapeFromBackup(choosedChinese,chineses,topN=topN)
        if len(resultChineses)==0:
            return None
        return resultChineses[random.randint(0,len(resultChineses)-1)]

    def chooseChineseWithShape(self,text,topN=10):
        topChoosedWord={}
        temp_topn=[0 for i in range(topN)]
        for word in text:
            if isChinese(word):
                choosedChinese = word
                simShapeChineses=self.chinese_shape_util.getAllSimilarityShape(choosedChinese)
                for index,topn_len in enumerate(temp_topn):
                    if len(simShapeChineses)>topn_len:
                        temp_topn[index]=len(simShapeChineses)
                        topChoosedWord[index]=(word,simShapeChineses)
                        break

        if len(topChoosedWord)==0:
            return None,None
        # topN中随机
        tuple=topChoosedWord[random.randint(0, min(topN, len(topChoosedWord) - 1))]
        return tuple[0],tuple[1]

    # def chooseChineseWithPinyinAndShape(self,choosedChinese):
    #     # 音近:从音近列表随机选一个拼音,根据拼音获取常用汉字列表
    #     simChineses=self.pinyin_util.getSimilarityChineseBySimPinyin(choosedChinese)
    #     if len(simChineses)==0:
    #         return None
    #     # 随机常用汉字
    #     return simChineses[random.randint(0,len(simChineses)-1)]

    def getNewText(self,text, pos, simChinese):
        if pos==0:
            return simChinese+text[pos+1:]
        elif pos==len(text)-1:
            return text[:-1]+simChinese
        else:
            return text[:pos]+simChinese+text[pos+1:]

    def data_generator(self,fieldname='correct_text',confusionPercent=30,simPinyinPercent=20,equalPinyinPercent=30,shapePercent=5,generateOnePercent=90,generateMaxForOne=3,seed=100):
        text_gens=[]
        # 分词
        # thu1 = thulac(seg_only=True)  # 只进行分词，不进行词性标注
        for ins in tqdm(self.train_data[:]):
            text=ins[fieldname]
            if text==None:
                continue
            if len(text)<1:
                continue
            iter=1
            chooseChoice=random.randint(0,100)
            if chooseChoice>=generateOnePercent:
                iter=random.randint(1,generateMaxForOne)
            for i in range(iter):
                # todo 查找与此相关混淆集词语:替换成音近形近字
                chooseWay = random.randint(0, 100)
                src_words = self.jieba.lcut(text)
                fine_words, single_grans = filterSingleWord(src_words)
                if chooseWay<confusionPercent:
                    fine_words,single_grans = filterSingleWord(src_words)
                    if len(fine_words) == 0:
                        continue
                    text, detail = self.proper.proper_gen(text,fine_words,seed=seed)
                    print("Choosed sim group:",text,detail)
                else:
                    pos=-1
                    # 优先选择易错词
                    choosedChinese,simChinese=self.chooseConfusionWordIfExist(fine_words,random)
                    if simChinese:
                        # 选择易错词，其次选择常用词
                        choosedChinese=self.chooseEasyErroWord(text)
                        if choosedChinese==None:
                            continue
                        # 80%:同音且形近中选
                        p = random.randint(0, 100)
                        if p < 60:
                            simChinese = self.chooseChineseWithPinyinAndShape(choosedChinese)
                            if simChinese == None:
                                continue
                        elif p>=60 and p<=90:
                            # 选择形近较多的字进行替换：此类字越多越可能混淆
                            choosedChinese,sim_chineses=self.chooseChineseWithShape(text)
                            if choosedChinese==None:
                                continue
                            simChinese=sim_chineses[random.randint(0,min(3,len(sim_chineses)-1))]
                        else:
                            chineses = self.pinyin_util.getSimilarityChineseBySimPinyin(choosedChinese)
                            if len(chineses) == 0:
                                continue
                            freq_chinese = self.filterNoFreqChinese(chineses)
                            if len(freq_chinese) == 0:
                                continue
                            simChinese=chineses[random.randint(0, len(freq_chinese) - 1)]
                    if choosedChinese == None:
                        continue
                    pos=text.find(choosedChinese)
                    if pos==-1:
                        continue
                    text=self.getNewText(text,pos,simChinese)
                    print("Choosed pinyin&shape:",(choosedChinese,simChinese),text)
            text_gens.append({
                "id": 1,
                "source": text,
                "target": ins[fieldname],
                "type": "negative"
            })
        return text_gens

    def filterNoFreqChinese(self, chineses):
        freq_chineses = self.pinyin_util.getSimilarityChineseByPinyin(chineses,doct_type=3)
        freq_words=[]
        for word in chineses:
            if word in freq_chineses:
                freq_words.append(word)
        return freq_words

    def chooseConfusionWordIfExist(self,fine_words,random):
        for word in fine_words:
            if word in self.confusions:
                err_words=self.confusions[word]
                return word,err_words[random.randint(0,len(err_words)-1)]
        return None,None

    def chooseEasyErroWord(self, text):
        for word in text:
            if word in self.easyerr_words:
                return word
        # 选最常用近音词表中替换
        pos=-1
        for i in range(5):
            pos = random.randint(0, len(text) - 1)
            if isChinese(text[pos]):
                break
        choosedChinese=None
        if pos!=-1:
            choosedChinese = text[pos]
        return choosedChinese

if __name__ == '__main__':
    basePath = "models/macbert/output"
    dg=DataGegerator(123,'preliminary_train_spell.json',basePath=basePath)
    outFile = os.path.join(basePath,'preliminary_train_spell_gen.json')
    # 小于4字长度的相似词组=confusionPercent，音近也形近且常用汉字选择=1-confusionPercent
    text_gens=dg.data_generator(confusionPercent=40)
    json.dump(text_gens, open(os.path.join(get_project_path(),outFile), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)