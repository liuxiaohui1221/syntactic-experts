
import json
import os.path
import random

import jieba
import numpy as np
from tqdm import tqdm

from ProjectPath import get_project_path
from knowledgebase.chinese_pinyin_util import ChinesePinyinUtil
from knowledgebase.chinese_shape_util import ChineseShapeUtil
from knowledgebase.dict.gen_confusion import readConfusions, readEasyConfusionWord, readEasyLossWord
from models.macbert.util.common import getEdits
from models.model_MiduCTC.src.baseline.ctc_vocab.config import VocabConf
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


def delChineseFromText(text, delChinese,del_all):
    if del_all:
        while True:
            pos = text.find(delChinese)
            if pos == -1:
                break
            text = text[:pos] + text[pos + 1:]
    else:
        pos = text.find(delChinese)
        if pos != -1:
            text = text[:pos] + text[pos + 1:]
    return text

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
        # 易loss词
        self.easyloss_words=readEasyLossWord()
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

    def data_generator(self,fieldname='correct_text',replaceAsPostive=False,lossPercent=20,swapPercent=5,replacePercent=15,
                       confusionInRepPercent=20,iter=1,seed=100):
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
            chooseWay = random.randint(0, 100)
            # if chooseChoice>=generateOnePercent:
            #     iter=random.randint(1,generateMaxForOne)
            loss_update=False
            for i in range(iter):
                src_words = self.jieba.lcut(text)
                fine_words, single_grans = filterSingleWord(src_words)
                newtext=text
                if chooseChoice<lossPercent:
                    loss_update=True
                    del_all = False
                    if chooseWay % 10 <= 2:
                        del_all = True
                    if chooseWay<30:
                        # 单字词
                        if len(single_grans)<1:
                            continue
                        elif len(single_grans)==1:
                            delChinese=single_grans[0]
                        else:
                            # 50%选择易loss词
                            # delChinese=self.chooseEasyLossIfExists(single_grans)
                            # if delChinese==None:
                            delChinese=single_grans[random.randint(0, len(single_grans) - 1)]
                        newtext=delChineseFromText(text,delChinese,del_all)
                    else:
                        if len(fine_words)==0:
                            continue
                        del_words=fine_words[random.randint(0,len(fine_words)-1)]
                        delChinese=None
                        if len(del_words)>0:
                            for i in range(5):
                                # # 50%选择易loss词
                                # delChinese = self.chooseEasyLossIfExists(single_grans)
                                # if delChinese == None:
                                delChinese = del_words[random.randint(0, len(del_words) - 1)]
                                if isChinese(delChinese):
                                    break
                        if delChinese == None:
                            continue
                        newtext=delChineseFromText(text,delChinese,del_all)
                    # print("Choosed del word:", delChinese, text)
                elif chooseChoice>lossPercent and chooseChoice<lossPercent+replacePercent:
                    # 查找与此相关混淆集词语:替换成音近形近字
                    if chooseWay<confusionInRepPercent:
                        if len(fine_words) == 0:
                            continue
                        newtext, detail = self.proper.proper_gen(text,fine_words,seed=seed)
                        # print("Choosed sim group:",text,detail)
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
                        newtext=self.getNewText(text,pos,simChinese)
                        # print("Choosed pinyin&shape:",(choosedChinese,simChinese),text)
                elif chooseChoice > lossPercent + replacePercent and chooseChoice <= lossPercent + replacePercent + swapPercent:
                    # 乱序问题
                    if len(fine_words)==0:
                        continue
                    pos=random.randint(0, len(fine_words) - 1)
                    swap_words=list(src_words[pos])
                    if len(swap_words)>1:
                        r_pos=random.randint(0,len(swap_words)-2)
                        r_word=swap_words[r_pos]
                        if isChinese(r_word) and isChinese(swap_words[r_pos+1]):
                            swap_words[r_pos]=swap_words[r_pos+1]
                            swap_words[r_pos + 1]=r_word
                    elif len(swap_words)==1 and pos<len(src_words)-1 and isChinese(src_words[pos+1][0]):
                        if not isChinese(swap_words[0]) :
                            continue
                        temp=swap_words[0]
                        swap_words[0]=src_words[pos+1][0]
                        src_words[pos + 1]=temp+src_words[pos+1][1:]

                    src_words[pos]="".join(swap_words)
                    newtext = "".join(src_words)

                if replaceAsPostive and loss_update==False:
                    # replace负例当做正例
                    ins[fieldname]=newtext
                text_gens.append({
                    "id": 1,
                    "source": newtext,
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

    def chooseEasyLossIfExists(self, grans):
        for word in grans:
            if word in self.easyloss_words:
                return word
        return None

    def takeLossDataFromTrain(self):
        inpath='models/model_MiduCTC/data/preliminary_a_data/preliminary_train.json'
        dicts = json.load(open(os.path.join(get_project_path(), inpath), encoding='utf-8'))
        filterd_data=[]
        for row in dicts:
            edits = getEdits(row['source'], row['target'])
            for edit in edits:
                if edit[0] == 'insert':
                    filterd_data.append(row)
                    break
        print("Got size", len(filterd_data))
        outpath = os.path.join(get_project_path(), 'models/model_MiduCTC/data/preliminary_a_data/loss_train.json')
        json.dump(filterd_data, open(outpath, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=4)
if __name__ == '__main__':
    basePath = "models/model_MiduCTC/data/preliminary_a_data"
    seed=1
    lossPercent=100
    dg=DataGegerator(seed,'preliminary_train.json',basePath=basePath)
    outFile = os.path.join(basePath,f'preliminary_train_gen_loss{lossPercent}_{seed}.json')
    # 缺字40%，replace:40%，剩余为正样本
    text_gens=dg.data_generator(fieldname='target',iter=5,lossPercent=lossPercent,swapPercent=0,replacePercent=0)
    print("saved",outFile,len(text_gens))
    json.dump(text_gens, open(os.path.join(get_project_path(),outFile), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=4)
    # 过滤真实缺字样本
    # dg.takeLossDataFromTrain()