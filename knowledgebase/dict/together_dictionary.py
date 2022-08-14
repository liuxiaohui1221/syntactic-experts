import os

from ProjectPath import get_project_path
from knowledgebase.dict.stop_check_word import readWordFile
from models.mypycorrector.utils.text_utils import is_chinese


def together_third_dict(inPath,outPath):
    dicts=[]
    dictPaths=[]
    for fn in os.listdir(inPath):
        print(fn)
        if fn[-3:]=='txt' or fn[-3:]=='dic':
            dictPaths.append(os.path.join(inPath,fn))
    for dictP in dictPaths:
        with open(dictP,"r",encoding="utf-8") as f:
            lines=f.readlines()
            for line in lines:
                temp_line=line.strip()
                if len(temp_line)<=1:
                    continue
                word=line.split(sep=' ')[0]
                word = word.split(sep='\t')[0]
                flag=False
                for w in word:
                    if is_chinese(w)==False:
                        flag=True
                        break
                if flag:
                    continue
                dicts.append(word)
    sets=set(dicts)
    print("dict size:",len(sets))
    with open(outPath,"w",encoding="utf-8") as f:
        for word in sets:
            f.write(word)
# together_third_dict('third',os.path.join(get_project_path(),'knowledgebase/dict/custom_dict.txt'))

def together_core_proper(dictPaths,outPath):
    dicts=[]
    for dictP in dictPaths:
        with open(os.path.join(get_project_path(),dictP),"r",encoding="utf-8") as f:
            lines=f.readlines()
            for line in lines:
                temp_line=line.strip()
                if len(temp_line)<=1:
                    continue
                word=line.split(sep=' ')[0]
                word = word.split(sep='\t')[0]
                dicts.append(word)
    sets=set(dicts)
    print("dict size:",len(sets))
    with open(os.path.join(get_project_path(),outPath),"w",encoding="utf-8") as f:
        for word in sets:
            f.write(word+'\n')

# dictPaths=['knowledgebase/dict/third/ChengYu_Corpus5W.txt','knowledgebase/dict/third/THUOCL_poem.txt']
# together_core_proper(dictPaths,'knowledgebase/dict/chengyu.txt')

def together_file(dictPaths,outPath):
    dicts = []
    # 过滤实际检查为高区分度的成语
    checked_path = 'knowledgebase/dict/checked_chengyu.txt'
    checked_words = readWordFile(checked_path)
    for dictP in dictPaths:
        with open(os.path.join(get_project_path(), dictP), "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                temp_line = line.strip()
                if len(temp_line) <= 1:
                    continue
                chengyu=temp_line.split(sep='\t')[1]
                if chengyu in checked_words:
                    continue
                dicts.append(temp_line)

    sets = set(dicts)
    print("dict size:", len(sets))
    with open(os.path.join(get_project_path(), outPath), "w", encoding="utf-8") as f:
        for word in sets:
            f.write(word+'\n')

dictPaths=['knowledgebase/dict/maybe_badword_dict1.txt','knowledgebase/dict/maybe_badword_dict_val.txt','knowledgebase/dict/maybe_badword_dict_extend.txt']
together_file(dictPaths,'knowledgebase/dict/low_chengyu.txt')

def clear_bad_wrod():
    pass