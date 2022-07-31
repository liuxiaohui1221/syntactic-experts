import os

from ProjectPath import get_project_path


def together_third_dict(inPath,outPath):
    dicts=[]
    dictPaths=[]
    for fn in os.listdir(inPath):
        print(fn)
        if fn[-3:]=='dic':
            dictPaths.append(os.path.join(inPath,fn))
    for dictP in dictPaths:
        with open(dictP,"r",encoding="utf-8") as f:
            lines=f.readlines()
            for word in lines:
                dicts.append(word)
    sets=set(dicts)
    print("dict size:",len(sets))
    with open(outPath,"w",encoding="utf-8") as f:
        for word in sets:
            f.write(word)
together_third_dict('third',os.path.join(get_project_path(),'knowledgebase/dict/custom_dict.dic'))