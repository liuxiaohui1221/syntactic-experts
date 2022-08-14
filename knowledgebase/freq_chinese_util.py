import os.path

from ProjectPath import get_project_path


def gen_freq_file(inpath,outpath,sep=' '):
    freq_words=[]
    with open(os.path.join(get_project_path(),inpath),'r',encoding='utf-8') as f:
        lines=f.readlines()
        for line in lines:
            if len(line.strip())==0:
                continue
            words=line.strip().split(sep=sep)
            for word in words:
                freq_words.append(word.strip('\n'))
    with open(os.path.join(get_project_path(),outpath),'w',encoding='utf-8') as f:
        for word in set(freq_words):
            f.write(word+'\n')
    print("Saved words:",len(freq_words))
gen_freq_file('knowledgebase/data/top_freq_chinese_original.txt','knowledgebase/data/top_freq_chinese.txt')