import os

from ProjectPath import get_project_path


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

def readWordFile(inPath,take=0):
    words = []
    with open(os.path.join(get_project_path(), inPath), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            temp_line = line.strip()
            if len(temp_line) <= 1:
                print(temp_line)
                continue
            if take==0:
                words.append(temp_line)
            else:
                word=temp_line.split('\t')[take]
                words.append(word)
    return words
def clear_stop_proper(inPath='knowledgebase/dict/fine_chengyu.txt',
                  outPath='knowledgebase/dict/fine_chengyu.txt'):
    badword_path='knowledgebase/dict/maybe_badword_dict_train1.txt'
    checked_path='knowledgebase/dict/checked_chengyu.txt'
    before_words = readWordFile(inPath)
    bad_words = readWordFile(badword_path,take=1)
    checked_words=readWordFile(checked_path)
    print("bad_words:",set(bad_words))
    total=0
    # 过滤无区分度词或错词
    filtered_words=[]
    for word in before_words:
        if word in bad_words:
            continue
        filtered_words.append(word)

    # 添加已检测实际为正例成语
    filtered_words.extend(checked_words)
    with open(os.path.join(get_project_path(), outPath), 'w', encoding='utf-8') as f:
        for word in filtered_words:
            if len(word)<1:
                continue
            f.write(word + '\n')
            total+=1
    print("Filterd before and after words:",len(before_words),total,outPath)


clear_stop_proper()