import os

from tqdm import tqdm

from ProjectPath import get_project_path
from knowledgebase.tencent.SentenceSimilarity import WordSentenceSimliarity
from models.ECSpell.Code.ProjectPath import get_ecspell_path


def gen_stop_word(inPath='knowledgebase/dict/freq_word.txt',
                  outPath='knowledgebase/dict/proper_stopwords.txt'):
    words = []
    with open(os.path.join(get_project_path(), inPath), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            temp_line = line.strip()
            if len(temp_line) <= 1:
                print(temp_line)
                continue
            words.append(temp_line)
    with open(os.path.join(get_project_path(), outPath), 'w', encoding='utf-8') as f:
        for word in words:
            f.write(word+'\n')
# gen_stop_word()

def readWordFile(inPath,take=-1):
    words = []
    with open(os.path.join(get_project_path(), inPath), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            temp_line = line.strip()
            if len(temp_line) <= 1:
                print(temp_line)
                continue
            if take==-1:
                words.append(temp_line)
            elif take==0:
                wordtuple = temp_line.split('\t')
                words.append(wordtuple)
            else:
                word=temp_line.split('\t')[take]
                words.append(word)
    return words

def clear_or_extend_proper(inPath='knowledgebase/dict/chengyu.txt',
                  outPath='knowledgebase/dict/fine_chengyu.txt',filter_low_proper=False):
    wss=WordSentenceSimliarity()
    # 通过腾讯词向量检测是否为专有词，是则扩充词汇，否则，去除对应区分度低的专名词
    maybe_badword_path = 'knowledgebase/dict/low_chengyu.txt'
    stop_proper_path='knowledgebase/dict/stop_proper.txt'
    out_low_proper_path='knowledgebase/dict/extend_low_proper.txt'
    checked_proper_path='knowledgebase/dict/checked_chengyu.txt'

    fine_words = readWordFile(inPath)
    low_proper_words = readWordFile(stop_proper_path)
    maybe_low_proper_words = readWordFile(maybe_badword_path,take=0)
    checked_words=readWordFile(checked_proper_path)
    # 添加已检测实际为正例成语
    fine_words.extend(checked_words)
    print("maybe bad_words:",len(maybe_low_proper_words))

    # 过滤无区分度词或错词
    for word_tuple in tqdm(maybe_low_proper_words):
        if not wss.existTencentWord(word_tuple[0]):
            low_proper_words.append("\t".join(word_tuple))
            continue
    new_fine_words=[]
    if filter_low_proper:
        for f_word in fine_words:
            if f_word in low_proper_words:
                continue
            new_fine_words.append(f_word)
        new_fine_words = set(new_fine_words)
        with open(os.path.join(get_project_path(), outPath), 'w', encoding='utf-8') as f:
            for word in new_fine_words:
                if len(word) < 1:
                    continue
                f.write(word.strip('\n') + '\n')
    stop_words=set(low_proper_words)
    with open(os.path.join(get_project_path(), out_low_proper_path), 'w', encoding='utf-8') as f:
        for word in stop_words:
            if len(word) < 1:
                continue
            f.write(word.strip('\n') + '\n')

    print("Filterd [fine words, bad proper words]:",len(fine_words),len(stop_words),outPath)

def together_maybe_badword(inPath,outPath=None):
    dictPaths=[os.path.join(get_project_path(),outPath)]
    for fn in os.listdir(inPath):
        if fn[:5]=='maybe':
            print(fn)
            dictPaths.append(os.path.join(inPath,fn))
    dicts=[]
    for dictP in dictPaths:
        with open(dictP,"r",encoding="utf-8") as f:
            lines=f.readlines()
            for line in lines:
                temp_line=line.strip('\n')
                if len(temp_line)<=1:
                    continue
                dicts.append(temp_line)
    sets=set(dicts)

    print("dict size:",len(sets))
    with open(os.path.join(get_project_path(),outPath),"w",encoding="utf-8") as f:
        for word in sets:
            f.write(word.strip('\n')+'\n')
def unique_file(inpath='knowledgebase/dict/extend_low_chengyu.txt'):
    bad_words = readWordFile(inpath)
    total=0
    with open(os.path.join(get_project_path(), inpath), 'w', encoding='utf-8') as f:
        for word in set(bad_words):
            if len(word)<1:
                continue
            f.write(word.strip('\n') + '\n')
            total+=1
    print("Before,after:",len(bad_words),total)

# clear_or_extend_proper(inPath='models/mypycorrector/data/extend_chengyu.txt',
#                   outPath='knowledgebase/dict/extend_chengyu.txt')
# together_maybe_badword(os.path.join(get_project_path(), 'knowledgebase/dict'),
#                        outPath='knowledgebase/dict/extend_low_chengyu.txt')
# unique_file()
# unique_file(inpath='knowledgebase/dict/low_chengyu.txt')
# clear_or_extend_proper()
# unique_file(inpath='knowledgebase/dict/confusions.txt')