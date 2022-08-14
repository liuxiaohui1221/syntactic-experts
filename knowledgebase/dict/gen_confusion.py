import os.path
import re
from collections import defaultdict

from pypinyin import pinyin, Style

from ProjectPath import get_project_path
from knowledgebase.chinese_shape_util import ChineseShapeUtil
from knowledgebase.tencent.SentenceSimilarity import WordSentenceSimliarity
from models.mypycorrector.proper_corrector import ProperCorrector
from models.mypycorrector.utils.text_utils import is_chinese


def readAndParse(inpath):
    word_pairs=[]
    with open(os.path.join(get_project_path(),inpath),'r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            temp_line=line.strip()
            if len(temp_line)<=1:
                continue
            arr1=line.split(sep='.')
            if len(arr1)<=1:
                continue
            arr2=re.split('（|\(',arr1[1])
            word=arr2[0]
            word_confusion=re.split('）|\)',arr2[1])[0]
            print(word,word_confusion)
            word_pairs.append([word,word_confusion])
    return word_pairs
def readAndParseTwoWord(inpath):
    word_pairs=[]
    with open(os.path.join(get_project_path(),inpath),'r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            temp_line=line.strip()
            if len(temp_line)<=1:
                continue
            arr1 = re.split(' |\t',line)
            if len(arr1) <= 1:
                continue
            for ar in arr1:
                arr2=re.split('（|\(',ar)
                word=arr2[0]
                word_confusion=re.split('）|\)',arr2[1])[0]
                print(word,word_confusion)
                word_pairs.append([word,word_confusion])
    return word_pairs

def saveToFile(outPath='knowledgebase/data/confusion/format_confusion_word.txt',inPath='knowledgebase/data/confusion/confusion_word.txt'):
    word_pairs=readAndParse(inPath)
    with open(os.path.join(get_project_path(), outPath), 'w', encoding='utf-8') as f:
        for confusion_word in word_pairs:
            f.write("_".join(confusion_word)+"\n")

def saveTwoWordToFile(outPath='knowledgebase/data/confusion/format_confusion_twoword.txt',
                      inPath='knowledgebase/data/confusion/two_word_confusion.txt'):
    word_pairs=readAndParseTwoWord(inPath)
    with open(os.path.join(get_project_path(), outPath), 'w', encoding='utf-8') as f:
        for confusion_word in word_pairs:
            f.write("_".join(confusion_word)+"\n")

# saveToFile()
# saveTwoWordToFile()

def readAndFormatConfusion(saveMode=1,thresh_score=0.8,outPath='models/mypycorrector/data/confusion_pair.txt',
                           inPath='knowledgebase/data/confusion/format_confusion_word.txt'):
    confusion_pairs=[]
    pc=ProperCorrector()
    shapeUtil=ChineseShapeUtil()
    wss = WordSentenceSimliarity()
    with open(os.path.join(get_project_path(),inPath),'r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            line=line.strip('\n')
            if len(line)<=1:
                continue
            confusion_pair=line.strip('\n').split(sep='_')
            # 找到正确字合适替换位置：近音
            ch_pinyins = pinyin(confusion_pair[0], style=Style.TONE3)
            if len(confusion_pair[1])>1:
                confusion_pair[1]=confusion_pair[1][0]
            right_pinyins = pinyin(confusion_pair[1], style=Style.TONE3, heteronym=True)[0]
            if is_chinese(confusion_pair[1])==False:
                continue
            poss=[]
            for index,word_pyins in enumerate(ch_pinyins):
                for word_pyin in word_pyins:
                    if word_pyin in right_pinyins:
                        poss.append(index)
                        break
            max_score_pos = -1
            max_score = -1
            if len(poss)==0:
                # 使用形近度查找位置
                flag=False

                for index, word in enumerate(confusion_pair[0]):
                    if saveMode==1:
                        sim_score = pc.get_word_similarity_score(word, confusion_pair[1])
                    else:
                        sim_score = shapeUtil.getShapeSimScore(word,confusion_pair[1])
                    if max_score<sim_score:
                        max_score=sim_score
                        max_score_pos=index
                    if sim_score > thresh_score:
                        poss.append(index)
                        flag=True
                        # print("high score word:", sim_score, word, confusion_pair[1])
                if flag==False:
                    print("Not find word:",confusion_pair)
            right_word=confusion_pair[0]

            if saveMode==1:
                for pos in poss:
                    right_word=right_word[:pos]+confusion_pair[1]+right_word[pos+1:]
                # 过滤无区分度的词:错词也是正常词
                # if wss.existTencentWord(confusion_pair[0]):
                #     continue
                confusion_pairs.append(confusion_pair[0] + '\t' + right_word)
            else:
                if len(poss)==1:
                    for pos in poss:
                        right_word=right_word[:pos]+confusion_pair[1]+right_word[pos+1:]
                else:
                    right_word = right_word[:max_score_pos] + confusion_pair[1] + right_word[max_score_pos + 1:]
                # 过滤无区分度的词:错词也是正常词
                # if wss.existTencentWord(right_word):
                #     print("Filter low confusion:",right_word,confusion_pair[0])
                #     continue
                confusion_pairs.append(right_word + '\t' + confusion_pair[0])
    with open(os.path.join(get_project_path(),outPath),'w',encoding='utf-8') as f:
        for pair in confusion_pairs:
            f.write(pair+'\n')

def readConfusions(inPath='knowledgebase/dict/confusions.txt'):
    confusion_pairs = defaultdict(list)
    with open(os.path.join(get_project_path(),inPath),'r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            if len(line.strip())==0:
                continue
            err_right=line.strip('\n').split(sep='\t')
            confusion_pairs[err_right[1]].extend(err_right[0])
    conf_len=len(confusion_pairs)
    print("confusions:",conf_len)
    return confusion_pairs
# readAndFormatConfusion()
# readAndFormatConfusion(saveMode=2,thresh_score=0.1,outPath='models/mypycorrector/data/confusion_twoword_pair.txt',
#                            inPath='knowledgebase/data/confusion/format_confusion_twoword.txt')