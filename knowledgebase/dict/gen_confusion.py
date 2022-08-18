import json
import os.path
import re
from collections import defaultdict

from pypinyin import pinyin, Style

from ProjectPath import get_project_path
from knowledgebase.chinese_pinyin_util import ChinesePinyinUtil
from knowledgebase.chinese_shape_util import ChineseShapeUtil
from knowledgebase.tencent.SentenceSimilarity import WordSentenceSimliarity
from models.macbert.util.common import getSpellErrorWord
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
                # 过滤无区分度的词：也是专名词
                if wss.existTencentWord(confusion_pair[0]):
                    continue
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

def saveToConfusion(inPath='knowledgebase/data/confusion/easy_wrong_words.json'):
    outPath='knowledgebase/dict/confusions-extend.txt'
    confusion_pairs=json.load(
        open(os.path.join(get_project_path(), inPath),
             encoding='utf-8'))
    with open(os.path.join(get_project_path(), outPath), 'w', encoding='utf-8') as f:
        for key,val in confusion_pairs.items():
            f.write(key+'\t'+ val + '\n')
def readConfusions(inPath='knowledgebase/dict/confusions.txt'):
    confusion_pairs = defaultdict(list)
    with open(os.path.join(get_project_path(),inPath),'r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            if len(line.strip())==0:
                continue
            err_right=line.strip('\n').split(sep='\t')
            if len(err_right)==1:
                err_right = line.strip('\n').split(sep=' ')
            if len(err_right)<2:
                continue
            confusion_pairs[err_right[1]].extend(err_right[0])
    conf_len=len(confusion_pairs)
    print("confusions:",conf_len)
    return confusion_pairs
# readAndFormatConfusion()
# readAndFormatConfusion(saveMode=2,thresh_score=0.1,outPath='models/mypycorrector/data/confusion_twoword_pair.txt',
#                            inPath='knowledgebase/data/confusion/format_confusion_twoword.txt')


def saveToSingleWordConfusion(inPath='knowledgebase/data/confusion/low_confusion.txt'):
    outPath = 'knowledgebase/confusion/confusion_low_singleword.txt'
    confusion_pairs = []
    with open(os.path.join(get_project_path(), inPath), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if len(line.strip()) == 0:
                continue
            arr=eval(line.strip('\n'))
            # 取replace的
            for pair in arr:
                if pair[0]!='replace':
                    continue
                confusion_pairs.append(pair[1] + '\t' + pair[2])

    with open(os.path.join(get_project_path(), outPath), 'w', encoding='utf-8') as f:
        for word in set(confusion_pairs):
            f.write(word + '\n')
def readEasyConfusionWord():
    confusion_path = os.path.join(get_project_path(), 'models/mypycorrector/data/confusion_pair.txt')
    single_confusion_path = os.path.join(get_project_path(), 'knowledgebase/confusion/confusion_low_singleword.txt')
    inpaths=[confusion_path,single_confusion_path]
    words=[]
    for inPath in inpaths:
        with open(os.path.join(get_project_path(), inPath), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip()) == 0:
                    continue
                arr=line.strip('\n').split(sep='\t')
                if len(arr)==1:
                    arr = line.strip('\n').split(sep=' ')
                if len(arr)<2:
                    continue
                s_words,t_words=getSpellErrorWord(arr[0],arr[1])
                s_words.extend(t_words)
                for w_tuple in s_words:
                    if is_chinese(w_tuple[0]) == False:
                        continue
                    words.append(w_tuple[0])
    uniqueWords=set(words)
    print("easy words:",len(uniqueWords))
    return uniqueWords
def checkErrWordProperInDB(word,proper_names,wss):
    if wss.existTencentWord(word) and proper_names.get(word,None)!=None:
        return True
    return False

def load_proper(proper_name_path):
    words={}
    with open(proper_name_path, 'r', encoding='utf-8') as f:
        for w in f:
            w = w.strip()
            if w.startswith('#'):
                continue
            if w:
                words[w]=w
    return words


def saveToMultiLowWordConfusion(inPaths):
    # 将混淆集分为可区分混淆集和不易区分混淆集
    confusion_pairs=[]
    confusion_low=[]
    pair_outpath='knowledgebase/confusion/good_confusions.txt'
    low_outpath='knowledgebase/confusion/low_confusions.txt'
    wss = WordSentenceSimliarity()
    proper_name_path = os.path.join(get_project_path(), 'knowledgebase/dict/custom_dict.txt')
    proper_names = load_proper(proper_name_path)
    for inpath in inPaths:
        with open(os.path.join(get_project_path(), inpath), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip()) == 0:
                    continue
                err_right=line.strip('\n').split(sep='\t')
                flag = checkErrWordProperInDB(err_right[0],proper_names,wss)
                if flag:
                    confusion_low.append(line.strip('\n'))
                else:
                    confusion_pairs.append(line.strip('\n'))
    with open(os.path.join(get_project_path(), pair_outpath), 'w', encoding='utf-8') as f:
        for word in set(confusion_pairs):
            f.write(word + '\n')
    with open(os.path.join(get_project_path(), low_outpath), 'w', encoding='utf-8') as f:
        for word in set(confusion_low):
            f.write(word + '\n')
saveToSingleWordConfusion()
confusion_paths=['knowledgebase/confusion/confusion_pair.txt',
                 'knowledgebase/confusion/confusions.txt',
                'knowledgebase/confusion/confusion_twoword_pair.txt',
                'knowledgebase/confusion/confusions-extend.txt'
            ]
saveToMultiLowWordConfusion(confusion_paths)